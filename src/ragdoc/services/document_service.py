"""Document upload, listing and deletion. The UI's only entry point for documents."""

import hashlib
import logging
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from ragdoc.config import Settings, get_settings
from ragdoc.ingestion.parser import SUPPORTED_EXTENSIONS, UnsupportedFileTypeError
from ragdoc.ingestion.pipeline import (
    IngestionCancelled,
    IngestionPipeline,
    ProgressCallback,
    build_pipeline,
)
from ragdoc.schemas import ChunkingConfig, IngestionProgress
from ragdoc.storage.milvus_store import MilvusStore, get_milvus_store
from ragdoc.storage.postgres.models import DocumentStatus
from ragdoc.storage.postgres.repositories import DocumentRepository
from ragdoc.storage.postgres.session import session_scope

logger = logging.getLogger(__name__)

SessionScope = Callable[[], AbstractContextManager[Session]]

# Parsing uses every core it can get, so two ingestions at once only slow each other down.
# One lock for the whole process, whichever service instance is used.
_ingest_lock = threading.Lock()


class DocumentInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    status: DocumentStatus
    error: str | None = None
    n_text_chunks: int = 0
    n_table_chunks: int = 0
    n_image_chunks: int = 0
    ocr_used: bool = False
    # The chunking strategy used for the text, and its parameters.
    chunker: str = "recursive"
    chunk_params: dict[str, Any] | None = None
    created_at: datetime
    # How long ingestion took; None while processing or if it did not finish.
    ingest_seconds: float | None = None
    # Per-stage times of the ingestion that just finished (parse_ms, embed_ms, ...). Only set
    # on the value returned by `ingest`.
    timings: dict[str, float] = {}
    # Windows of text the LLM-based chunking could not handle and split plainly instead. Only
    # set on the value returned by `ingest`.
    chunk_fallbacks: int = 0


class DuplicateDocumentError(Exception):
    """The same file content has already been uploaded."""

    def __init__(self, existing: DocumentInfo):
        super().__init__(f"This file was already uploaded as '{existing.filename}'.")
        self.existing = existing


class DocumentService:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        store: MilvusStore | None = None,
        pipeline: IngestionPipeline | None = None,
        session_scope: SessionScope = session_scope,
    ):
        self._settings = settings or get_settings()
        self._store = store
        self._pipeline = pipeline
        self._session_scope = session_scope

    @property
    def supported_extensions(self) -> list[str]:
        """File extensions that can be uploaded, without the dot, sorted."""
        return sorted(extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS)

    @property
    def ocr_default(self) -> bool:
        return self._settings.ocr_default

    @property
    def chunking_defaults(self) -> dict[str, ChunkingConfig]:
        """Every chunking strategy with its configured default parameters, by name."""
        return self._settings.chunking_defaults()

    @property
    def default_chunker(self) -> str:
        """The strategy an upload uses unless it chooses another."""
        return self._settings.chunker

    # Built on first use, so listing documents never loads parsing or embedding models.
    @property
    def store(self) -> MilvusStore:
        if self._store is None:
            if self._settings is get_settings():
                self._store = get_milvus_store()
            else:
                self._store = MilvusStore(self._settings)
                self._store.ensure_collections()
        return self._store

    @property
    def pipeline(self) -> IngestionPipeline:
        if self._pipeline is None:
            self._pipeline = build_pipeline(self.store, self._settings)
        return self._pipeline

    def ingest(
        self,
        filename: str,
        data: bytes,
        on_progress: ProgressCallback | None = None,
        ocr: bool | None = None,
        chunking: ChunkingConfig | None = None,
    ) -> DocumentInfo:
        """Store and index an uploaded file. `ocr=None` and `chunking=None` mean the configured
        defaults.

        Returns the document with status `ready`, or `failed` with `error` set if processing
        went wrong (the failed document stays listed so the error is visible; uploading the
        same file again replaces it).

        Raises `UnsupportedFileTypeError` or `DuplicateDocumentError` before anything is stored,
        and `IngestionCancelled` if the document was deleted while it was being processed.
        """
        filename = Path(filename).name
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"{suffix or 'A file with no extension'} is not supported. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        ocr = self._settings.ocr_default if ocr is None else ocr
        chunking = chunking or self._settings.default_chunking()

        file_hash = hashlib.sha256(data).hexdigest()
        retry_of: uuid.UUID | None = None
        with self._session_scope() as session:
            existing = DocumentRepository(session).get_by_hash(file_hash)
            if existing is not None:
                if existing.status != DocumentStatus.FAILED:
                    raise DuplicateDocumentError(DocumentInfo.model_validate(existing))
                retry_of = existing.id
        if retry_of is not None:
            # Re-uploading a file whose earlier attempt failed replaces that attempt.
            self.delete(retry_of)

        document_id = uuid.uuid4()
        upload_path = self._upload_dir(document_id) / filename
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        upload_path.write_bytes(data)

        # Committed before processing starts, so the document is listed as `processing`.
        with self._session_scope() as session:
            DocumentRepository(session).create(
                document_id=document_id,
                filename=filename,
                file_hash=file_hash,
                file_path=upload_path.relative_to(self._settings.data_path).as_posix(),
                chunker=chunking.strategy,
                chunk_params=chunking.params,
            )

        def deleted() -> bool:
            return not self._exists(document_id)

        try:
            self._wait_for_turn(filename, on_progress, deleted)
            try:
                result = self.pipeline.run(
                    upload_path,
                    doc_id=str(document_id),
                    filename=filename,
                    ocr=ocr,
                    chunking=chunking,
                    on_progress=on_progress,
                    should_stop=deleted,
                )
            finally:
                _ingest_lock.release()
            if deleted():
                # Deleted while the last step was running: that step's output is orphaned.
                self.pipeline.cleanup(str(document_id))
                raise IngestionCancelled(f"Ingestion of {filename} was cancelled.")
        except IngestionCancelled:
            shutil.rmtree(self._upload_dir(document_id), ignore_errors=True)
            raise
        except Exception as exc:
            logger.exception("Ingestion failed for %s (%s)", filename, document_id)
            failed = self._mark_failed(document_id, f"{type(exc).__name__}: {exc}")
            if failed is None:
                # It failed and was deleted meanwhile: for the caller, a cancellation.
                shutil.rmtree(self._upload_dir(document_id), ignore_errors=True)
                raise IngestionCancelled(f"Ingestion of {filename} was cancelled.") from exc
            return failed
        except BaseException:
            # Not an error in the document: the caller went away (page refresh, closed tab,
            # Ctrl+C). The pipeline has already removed what it wrote.
            self._mark_failed(document_id, "The upload was interrupted. Upload the file again.")
            raise

        with self._session_scope() as session:
            documents = DocumentRepository(session)
            documents.add_ingestion_metrics(document_id, **result.timings)
            document = documents.mark_ready(
                document_id,
                n_text_chunks=result.n_text_chunks,
                n_table_chunks=result.n_table_chunks,
                n_image_chunks=result.n_image_chunks,
                ocr_used=result.ocr_used,
            )
            info = DocumentInfo.model_validate(document)
        info.ingest_seconds = result.timings.get("total_ms", 0) / 1000
        info.timings = result.timings
        info.chunk_fallbacks = result.n_chunk_fallbacks
        return info

    def list_documents(self) -> list[DocumentInfo]:
        """All documents, newest first."""
        with self._session_scope() as session:
            documents = DocumentRepository(session)
            seconds = documents.ingestion_seconds()
            infos = [DocumentInfo.model_validate(document) for document in documents.list_all()]
        for info in infos:
            info.ingest_seconds = seconds.get(info.id)
        return infos

    def delete(self, document_id: uuid.UUID | str) -> bool:
        """Remove a document everywhere: vectors, extracted images, the uploaded file and the
        registry row. Returns False if the document does not exist.

        The registry row goes last, so a delete that fails part-way can simply be repeated.
        If the document is still being processed, removing the row is what stops that work: the
        ingestion notices at its next checkpoint and removes anything it wrote in the meantime.
        """
        document_id = uuid.UUID(str(document_id))
        if not self._exists(document_id):
            return False

        self.store.delete_document(str(document_id))
        shutil.rmtree(self._settings.images_dir / str(document_id), ignore_errors=True)
        shutil.rmtree(self._upload_dir(document_id), ignore_errors=True)

        with self._session_scope() as session:
            return DocumentRepository(session).delete(document_id)

    def recover_interrupted(self) -> int:
        """Mark documents left in `processing` as failed and remove their partial data.

        For application start-up only: at that point nothing can be processing, so such a
        document was cut off by a restart. Returns how many were found.
        """
        with self._session_scope() as session:
            stuck = [document.id for document in DocumentRepository(session).list_processing()]
        for document_id in stuck:
            self.store.delete_document(str(document_id))
            shutil.rmtree(self._settings.images_dir / str(document_id), ignore_errors=True)
            self._mark_failed(
                document_id, "Processing was interrupted by a restart. Upload the file again."
            )
        return len(stuck)

    def _wait_for_turn(
        self, filename: str, on_progress: ProgressCallback | None, deleted: Callable[[], bool]
    ) -> None:
        """Take the ingestion lock, reporting the wait and giving up if the document is deleted."""
        started = time.perf_counter()
        while not _ingest_lock.acquire(timeout=1.0):
            if deleted():
                raise IngestionCancelled(f"Ingestion of {filename} was cancelled.")
            if on_progress is not None:
                on_progress(
                    IngestionProgress(
                        stage="wait",
                        message="Waiting for another upload to finish",
                        elapsed_s=time.perf_counter() - started,
                    )
                )

    def _exists(self, document_id: uuid.UUID) -> bool:
        with self._session_scope() as session:
            return DocumentRepository(session).get(document_id) is not None

    def _mark_failed(self, document_id: uuid.UUID, error: str) -> DocumentInfo | None:
        """Record a failure. Returns None if the document has been deleted meanwhile."""
        with self._session_scope() as session:
            documents = DocumentRepository(session)
            if documents.get(document_id) is None:
                return None
            return DocumentInfo.model_validate(documents.mark_failed(document_id, error))

    def _upload_dir(self, document_id: uuid.UUID) -> Path:
        return self._settings.uploads_dir / str(document_id)
