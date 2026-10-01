"""DocumentService against the real database, with the ingestion pipeline faked.

Covers what happens around an ingestion: cancelling by deleting, interruption, recovery after
a restart, and uploads taking turns. No parsing or embedding models are loaded.
"""

import uuid

import pytest

from ragdoc.config import Settings
from ragdoc.ingestion.pipeline import IngestionCancelled, IngestionResult
from ragdoc.schemas import LLMChunking, SemanticChunking
from ragdoc.services.document_service import DocumentService
from ragdoc.storage.postgres.models import DocumentStatus
from ragdoc.storage.postgres.repositories import DocumentRepository

pytestmark = pytest.mark.integration


class FakePipeline:
    """Runs `during(service, doc_id)` in place of the real work, then checks `should_stop`
    the way the real pipeline does at its checkpoints."""

    def __init__(self, during=None, error: BaseException | None = None):
        self.during = during
        self.error = error
        self.cleaned: list[str] = []
        self.ocr_seen: list[bool] = []
        self.chunking_seen: list = []
        self.n_chunk_fallbacks = 0
        self.service: DocumentService | None = None

    def run(
        self,
        path,
        *,
        doc_id,
        filename,
        ocr=False,
        chunking=None,
        on_progress=None,
        should_stop=None,
    ):
        self.ocr_seen.append(ocr)
        self.chunking_seen.append(chunking)
        try:
            if self.during:
                self.during(self.service, doc_id)
            if self.error:
                raise self.error
            if should_stop is not None and should_stop():
                raise IngestionCancelled(f"Ingestion of {filename} was cancelled.")
        except BaseException:
            self.cleanup(doc_id)
            raise
        return IngestionResult(
            n_text_chunks=3,
            ocr_used=ocr,
            chunking=chunking,
            n_chunk_fallbacks=self.n_chunk_fallbacks,
            timings={"parse_ms": 1, "chunk_ms": 1, "embed_ms": 1, "store_ms": 1, "total_ms": 4000},
        )

    def cleanup(self, doc_id):
        self.cleaned.append(doc_id)


@pytest.fixture
def make_service(tmp_path, scoped_sessions):
    def make(pipeline: FakePipeline, **settings) -> DocumentService:
        from unittest.mock import MagicMock

        service = DocumentService(
            Settings(_env_file=None, data_dir=tmp_path, **settings),
            store=MagicMock(),
            pipeline=pipeline,
            session_scope=scoped_sessions,
        )
        pipeline.service = service
        return service

    return make


def listed(service: DocumentService, *filenames: str) -> list:
    """The documents with these names. The database may hold others: tests run inside a
    rolled-back transaction on the real one."""
    return [d for d in service.list_documents() if d.filename in filenames]


def content() -> bytes:
    return f"unique {uuid.uuid4()}".encode()


def test_ready_document_records_ocr_choice_and_time(make_service):
    pipeline = FakePipeline()
    service = make_service(pipeline, ocr_default=True)

    by_default = service.ingest("a.txt", content())
    switched_off = service.ingest("b.txt", content(), ocr=False)

    assert pipeline.ocr_seen == [True, False]
    assert (by_default.ocr_used, switched_off.ocr_used) == (True, False)
    assert by_default.ingest_seconds == 4.0 and by_default.timings["total_ms"] == 4000
    listed = {d.filename: d for d in service.list_documents()}
    assert listed["a.txt"].ocr_used is True and listed["a.txt"].ingest_seconds == 4.0
    assert listed["b.txt"].ocr_used is False


def test_chunking_choice_is_passed_on_and_recorded(make_service):
    pipeline = FakePipeline()
    pipeline.n_chunk_fallbacks = 2
    service = make_service(pipeline, chunker="semantic", semantic_breakpoint_percentile=80)
    assert service.default_chunker == "semantic"
    chosen = LLMChunking(window_chars=3000)

    by_default = service.ingest("a.txt", content())
    by_choice = service.ingest("b.txt", content(), chunking=chosen)

    assert pipeline.chunking_seen == [SemanticChunking(breakpoint_percentile=80), chosen]
    assert by_choice.chunk_fallbacks == 2
    listed = {d.filename: d for d in service.list_documents()}
    assert listed["a.txt"].chunker == "semantic"
    assert listed["a.txt"].chunk_params["breakpoint_percentile"] == 80
    assert (listed["b.txt"].chunker, listed["b.txt"].chunk_params) == ("llm", chosen.params)
    assert by_default.chunker == "semantic" and listed["b.txt"].chunk_fallbacks == 0


def test_failed_document_still_shows_its_chunking_choice(make_service):
    service = make_service(FakePipeline(error=RuntimeError("model crashed")))
    failed = service.ingest("bad.pdf", content(), chunking=LLMChunking())
    assert (failed.status, failed.chunker) == (DocumentStatus.FAILED, "llm")


def test_deleting_a_processing_document_cancels_its_ingestion(make_service, tmp_path):
    def delete_it(service, doc_id):
        assert service.delete(doc_id) is True

    pipeline = FakePipeline(during=delete_it)
    service = make_service(pipeline)

    with pytest.raises(IngestionCancelled):
        service.ingest("big.pdf", content())

    assert listed(service, "big.pdf") == []
    assert len(pipeline.cleaned) == 1  # whatever was written after the delete is removed
    assert list((tmp_path / "uploads").iterdir()) == []


def test_document_deleted_during_the_last_step_is_cleaned_up(make_service, tmp_path, monkeypatch):
    pipeline = FakePipeline()
    service = make_service(pipeline)
    # The pipeline finishes normally; the row disappears just before the service checks.
    real_run = pipeline.run

    def run_then_delete(path, *, doc_id, **kwargs):
        result = real_run(path, doc_id=doc_id, **kwargs)
        service.delete(doc_id)
        return result

    monkeypatch.setattr(pipeline, "run", run_then_delete)

    with pytest.raises(IngestionCancelled):
        service.ingest("big.pdf", content())

    assert listed(service, "big.pdf") == []
    assert len(pipeline.cleaned) == 1
    assert list((tmp_path / "uploads").iterdir()) == []


def test_interrupted_upload_is_marked_failed_and_can_be_uploaded_again(make_service):
    class ScriptStopped(BaseException):
        pass

    data = content()
    service = make_service(FakePipeline(error=ScriptStopped()))
    with pytest.raises(ScriptStopped):
        service.ingest("report.pdf", data)

    [document] = listed(service, "report.pdf")
    assert document.status is DocumentStatus.FAILED
    assert "interrupted" in document.error

    retry = make_service(FakePipeline())
    again = retry.ingest("report.pdf", data)
    assert again.status is DocumentStatus.READY
    assert [d.id for d in listed(retry, "report.pdf")] == [again.id]


def test_failure_is_recorded_and_lock_is_released(make_service):
    service = make_service(FakePipeline(error=RuntimeError("parser crashed")))
    failed = service.ingest("bad.pdf", content())
    assert failed.status is DocumentStatus.FAILED
    assert failed.error == "RuntimeError: parser crashed"

    # A failure must not leave the ingestion lock held.
    assert make_service(FakePipeline()).ingest("ok.txt", content()).status is DocumentStatus.READY


def test_recover_interrupted_marks_stuck_documents_failed(make_service, scoped_sessions, tmp_path):
    service = make_service(FakePipeline())
    ready = service.ingest("fine.txt", content())
    with scoped_sessions() as session:
        stuck = DocumentRepository(session).create(
            filename="stuck.pdf", file_hash=uuid.uuid4().hex * 2, file_path="uploads/x/stuck.pdf"
        )
        stuck_id = stuck.id
    leftovers = tmp_path / "images" / str(stuck_id)
    leftovers.mkdir(parents=True)
    (leftovers / "image_001.png").write_bytes(b"png")

    assert service.recover_interrupted() == 1

    by_id = {d.id: d for d in service.list_documents()}
    assert by_id[stuck_id].status is DocumentStatus.FAILED
    assert "restart" in by_id[stuck_id].error
    assert by_id[ready.id].status is DocumentStatus.READY
    assert not leftovers.exists()
    service.store.delete_document.assert_called_once_with(str(stuck_id))
    assert service.recover_interrupted() == 0


def test_upload_waits_for_its_turn_and_reports_the_wait(make_service):
    from ragdoc.services import document_service

    service = make_service(FakePipeline())
    waits = []

    def on_progress(progress):
        waits.append(progress)
        if progress.stage == "wait" and document_service._ingest_lock.locked():
            document_service._ingest_lock.release()  # the "other upload" finishes

    # Another ingestion is running: it holds the process-wide lock.
    assert document_service._ingest_lock.acquire(timeout=5)
    try:
        done = service.ingest("second.txt", content(), on_progress=on_progress)
    finally:
        if document_service._ingest_lock.locked():
            document_service._ingest_lock.release()

    assert waits[0].stage == "wait"
    assert waits[0].message == "Waiting for another upload to finish"
    assert done.status is DocumentStatus.READY
    assert not document_service._ingest_lock.locked()


def test_deleting_a_waiting_upload_cancels_it_without_running(make_service):
    from ragdoc.services import document_service

    pipeline = FakePipeline()
    service = make_service(pipeline)

    def delete_while_waiting(progress):
        [document] = listed(service, "queued.txt")
        service.delete(document.id)

    assert document_service._ingest_lock.acquire(timeout=5)
    try:
        with pytest.raises(IngestionCancelled):
            service.ingest("queued.txt", content(), on_progress=delete_while_waiting)
    finally:
        document_service._ingest_lock.release()

    assert pipeline.ocr_seen == []  # the pipeline never started
    assert listed(service, "queued.txt") == []
