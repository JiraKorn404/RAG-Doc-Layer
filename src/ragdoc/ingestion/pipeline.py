"""The ingestion pipeline: parse -> chunk -> embed -> store, for one document."""

import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.embeddings import Embeddings

from ragdoc.config import Settings, get_settings
from ragdoc.ingestion.chunking.base import Chunker
from ragdoc.ingestion.chunking.factory import get_chunker
from ragdoc.ingestion.parser import DocumentParser, get_parser
from ragdoc.llm.embeddings import get_text_embedder
from ragdoc.llm.image_embeddings import ImageEmbedder, get_image_embedder
from ragdoc.schemas import ChunkType, IngestionProgress
from ragdoc.storage.milvus_store import MilvusStore

ProgressCallback = Callable[[IngestionProgress], None]
# Returns True when the ingestion should be abandoned (its document was deleted).
StopCheck = Callable[[], bool]


class IngestionError(RuntimeError):
    pass


class IngestionCancelled(Exception):  # noqa: N818 - a normal outcome, not an error
    """The ingestion was stopped on purpose. Everything it wrote has been removed."""


@dataclass
class IngestionResult:
    n_text_chunks: int = 0
    n_table_chunks: int = 0
    n_image_chunks: int = 0
    n_pages: int | None = None
    ocr_used: bool = False
    # parse_ms, chunk_ms, embed_ms, store_ms, total_ms
    timings: dict[str, float] = field(default_factory=dict)


class IngestionPipeline:
    def __init__(
        self,
        *,
        parser: DocumentParser,
        chunker: Chunker,
        text_embedder: Embeddings,
        image_embedder_factory: Callable[[], ImageEmbedder],
        store: MilvusStore,
        settings: Settings | None = None,
    ):
        self._parser = parser
        self._chunker = chunker
        self._text_embedder = text_embedder
        # A factory, so the image model is only loaded for documents that contain images.
        self._image_embedder_factory = image_embedder_factory
        self._store = store
        self._settings = settings or get_settings()

    def image_dir(self, doc_id: str) -> Path:
        return self._settings.images_dir / doc_id

    def run(
        self,
        path: Path,
        *,
        doc_id: str,
        filename: str,
        ocr: bool = False,
        on_progress: ProgressCallback | None = None,
        should_stop: StopCheck | None = None,
    ) -> IngestionResult:
        """Ingest one file.

        `should_stop` is asked between page batches, between embedding batches and before
        storing; when it returns True the run ends with `IngestionCancelled`. However the run
        ends without a result (failure, cancellation, or the caller being interrupted), everything
        written for `doc_id` is removed first.
        """
        try:
            return self._run(path, doc_id, filename, ocr, on_progress, should_stop)
        except BaseException:
            # BaseException, not Exception: Streamlit stops a script (page refresh, closed tab)
            # by raising one that is not an Exception, from inside the progress callback.
            self.cleanup(doc_id)
            raise

    def cleanup(self, doc_id: str) -> None:
        """Remove a document's vectors and extracted images."""
        self._store.delete_document(doc_id)
        shutil.rmtree(self.image_dir(doc_id), ignore_errors=True)

    def _run(
        self,
        path: Path,
        doc_id: str,
        filename: str,
        ocr: bool,
        on_progress: ProgressCallback | None,
        should_stop: StopCheck | None,
    ) -> IngestionResult:
        timings: dict[str, float] = {}
        started = time.perf_counter()
        n_pages: int | None = None

        def timed(name: str, since: float) -> float:
            now = time.perf_counter()
            timings[name] = (now - since) * 1000
            return now

        def checkpoint(
            stage: str, message: str, current: int | None = None, total: int | None = None
        ) -> None:
            """Stop here if asked to; otherwise report what is about to happen."""
            if should_stop is not None and should_stop():
                raise IngestionCancelled(f"Ingestion of {filename} was cancelled.")
            if on_progress is not None:
                on_progress(
                    IngestionProgress(
                        stage=stage,
                        message=message,
                        current=current,
                        total=total,
                        elapsed_s=time.perf_counter() - started,
                    )
                )

        def on_pages(first: int, last: int, total: int) -> None:
            nonlocal n_pages
            n_pages = total
            pages = f"page {first}" if first == last else f"pages {first}-{last}"
            # `current` counts finished pages, so the bar is at zero during the first batch.
            checkpoint("parse", f"Parsing {pages} of {total}", first - 1, total)

        checkpoint("parse", "Parsing document" + (" with OCR" if ocr else ""))
        elements = self._parser.parse(path, self.image_dir(doc_id), ocr=ocr, on_pages=on_pages)
        mark = timed("parse_ms", started)

        checkpoint("chunk", "Chunking")
        chunks = self._chunker.chunk(elements, doc_id=doc_id, metadata={"filename": filename})
        text_chunks = [chunk for chunk in chunks if chunk.chunk_type is not ChunkType.IMAGE]
        image_chunks = [chunk for chunk in chunks if chunk.chunk_type is ChunkType.IMAGE]
        if not chunks:
            hint = (
                ""
                if ocr
                else " If it is a scanned document or a picture of text, upload it again with "
                "the OCR option switched on."
            )
            raise IngestionError(f"No text, tables or images were found in this file.{hint}")
        mark = timed("chunk_ms", mark)

        batch_size = self._settings.embed_batch_size
        text_vectors: list[list[float]] = []
        for start in range(0, len(text_chunks), batch_size):
            checkpoint("embed", "Embedding text and table chunks", start, len(text_chunks))
            batch = text_chunks[start : start + batch_size]
            text_vectors.extend(
                self._text_embedder.embed_documents([chunk.content for chunk in batch])
            )

        image_vectors: list[list[float]] = []
        if image_chunks:
            checkpoint("embed", "Loading the image model", 0, len(image_chunks))
            image_embedder = self._image_embedder_factory()
            for start in range(0, len(image_chunks), batch_size):
                checkpoint("embed", "Embedding images", start, len(image_chunks))
                batch = image_chunks[start : start + batch_size]
                image_vectors.extend(
                    image_embedder.embed_images([Path(chunk.image_path) for chunk in batch])
                )
            # Stored relative to the data directory, so the data folder can be moved.
            for chunk in image_chunks:
                relative = Path(chunk.image_path).relative_to(self._settings.data_path)
                chunk.image_path = relative.as_posix()
        mark = timed("embed_ms", mark)

        checkpoint("store", "Storing vectors")
        self._store.insert_text_chunks(text_chunks, text_vectors)
        self._store.insert_image_chunks(image_chunks, image_vectors)
        timed("store_ms", mark)
        timed("total_ms", started)

        return IngestionResult(
            n_text_chunks=sum(c.chunk_type is ChunkType.TEXT for c in text_chunks),
            n_table_chunks=sum(c.chunk_type is ChunkType.TABLE for c in text_chunks),
            n_image_chunks=len(image_chunks),
            n_pages=n_pages,
            ocr_used=ocr,
            timings=timings,
        )


def build_pipeline(store: MilvusStore, settings: Settings | None = None) -> IngestionPipeline:
    settings = settings or get_settings()
    return IngestionPipeline(
        parser=get_parser(settings),
        chunker=get_chunker(settings),
        text_embedder=get_text_embedder(settings),
        image_embedder_factory=lambda: get_image_embedder(settings),
        store=store,
        settings=settings,
    )
