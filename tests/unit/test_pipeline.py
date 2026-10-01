from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ragdoc.config import Settings
from ragdoc.ingestion.chunking.recursive import RecursiveChunker
from ragdoc.ingestion.parser import (
    DoclingParser,
    ParsedElement,
    ParseTimeoutError,
    UnsupportedFileTypeError,
)
from ragdoc.ingestion.pipeline import IngestionCancelled, IngestionError, IngestionPipeline
from ragdoc.schemas import ChunkType, IngestionProgress, LLMChunking


class FakeParser:
    """Stands in for Docling. `page_batches` makes it report page batches like a PDF."""

    def __init__(self, elements=None, error: BaseException | None = None, page_batches=()):
        self.elements = elements
        self.error = error
        self.page_batches = page_batches
        self.ocr_seen: list[bool] = []

    def parse(self, path: Path, image_dir: Path, *, ocr=False, on_pages=None):
        self.ocr_seen.append(ocr)
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / "image_001.png"
        image_path.write_bytes(b"png")
        for first, last, total in self.page_batches:
            on_pages(first, last, total)
        if self.error:
            raise self.error
        if self.elements is not None:
            return self.elements
        return [
            ParsedElement(type=ChunkType.TEXT, content="some text " * 30, page=1),
            ParsedElement(type=ChunkType.TABLE, content="| a |\n| --- |\n| 1 |", page=1),
            ParsedElement(type=ChunkType.IMAGE, caption="Figure 1", page=2, image_path=image_path),
        ]


class FakeTextEmbedder:
    def __init__(self):
        self.batches: list[int] = []

    def embed_documents(self, texts):
        self.batches.append(len(texts))
        return [[0.0] for _ in texts]


class FakeImageEmbedder:
    def embed_images(self, paths):
        assert all(Path(path).is_absolute() and Path(path).exists() for path in paths)
        return [[1.0] for _ in paths]


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path, embed_batch_size=2)


def make_pipeline(settings, parser, store=None, text_embedder=None, chunker_factory=None):
    def small_chunks(_config):
        return RecursiveChunker(chunk_size=100, chunk_overlap=10, table_max_chars=500)

    return IngestionPipeline(
        parser=parser,
        chunker_factory=chunker_factory or small_chunks,
        text_embedder=text_embedder or FakeTextEmbedder(),
        image_embedder_factory=FakeImageEmbedder,
        store=store or MagicMock(),
        settings=settings,
    )


def run(pipeline, tmp_path, **kwargs):
    return pipeline.run(tmp_path / "f.pdf", doc_id="doc-1", filename="f.pdf", **kwargs)


def test_run_stores_all_chunk_types_and_reports_counts(settings, tmp_path):
    store, embedder = MagicMock(), FakeTextEmbedder()
    pipeline = make_pipeline(settings, FakeParser(), store, embedder)

    result = run(pipeline, tmp_path)

    assert result.n_text_chunks >= 2
    assert (result.n_table_chunks, result.n_image_chunks) == (1, 1)
    assert set(result.timings) == {"parse_ms", "chunk_ms", "embed_ms", "store_ms", "total_ms"}
    assert result.ocr_used is False and result.n_pages is None

    text_chunks, text_vectors = store.insert_text_chunks.call_args.args
    assert len(text_chunks) == len(text_vectors) == result.n_text_chunks + 1
    assert all(
        chunk.metadata == {"filename": "f.pdf", "chunker": "recursive"} for chunk in text_chunks
    )
    assert result.chunking == settings.default_chunking() and result.n_chunk_fallbacks == 0
    assert max(embedder.batches) <= 2 and sum(embedder.batches) == len(text_chunks)

    [image_chunk], image_vectors = store.insert_image_chunks.call_args.args
    assert image_chunk.image_path == "images/doc-1/image_001.png"
    assert image_vectors == [[1.0]]
    store.delete_document.assert_not_called()


def test_progress_reports_pages_then_each_stage(settings, tmp_path):
    events: list[IngestionProgress] = []
    parser = FakeParser(page_batches=[(1, 4, 9), (5, 8, 9), (9, 9, 9)])
    pipeline = make_pipeline(settings, parser)

    result = run(pipeline, tmp_path, on_progress=events.append)

    assert result.n_pages == 9
    assert [e.message for e in events[:4]] == [
        "Parsing document",
        "Parsing pages 1-4 of 9",
        "Parsing pages 5-8 of 9",
        "Parsing page 9 of 9",
    ]
    # `current` counts pages already done, so the fraction grows batch by batch.
    assert [(e.current, e.total) for e in events[1:4]] == [(0, 9), (4, 9), (8, 9)]
    assert events[0].fraction is None and events[2].fraction == pytest.approx(4 / 9)
    stages = [e.stage for e in events]
    assert stages == sorted(stages, key=["parse", "chunk", "embed", "store"].index)
    assert events[-1].message == "Storing vectors"
    embed = [e for e in events if e.message == "Embedding text and table chunks"]
    assert [e.current for e in embed] == list(range(0, embed[0].total, 2))
    assert all(a.elapsed_s <= b.elapsed_s for a, b in zip(events, events[1:], strict=False))


def test_ocr_choice_reaches_the_parser_and_the_result(settings, tmp_path):
    parser = FakeParser()
    events = []
    result = run(make_pipeline(settings, parser), tmp_path, ocr=True, on_progress=events.append)
    assert parser.ocr_seen == [True]
    assert result.ocr_used is True
    assert events[0].message == "Parsing document with OCR"


def test_stop_between_page_batches_cancels_and_cleans_up(settings, tmp_path):
    store = MagicMock()
    events = []
    parser = FakeParser(page_batches=[(1, 4, 12), (5, 8, 12), (9, 12, 12)])
    pipeline = make_pipeline(settings, parser, store)
    # Asked once at the start and once before each batch: stop before the second batch.
    answers = iter([False, False, True])

    with pytest.raises(IngestionCancelled):
        run(pipeline, tmp_path, on_progress=events.append, should_stop=lambda: next(answers))

    assert [e.message for e in events] == ["Parsing document", "Parsing pages 1-4 of 12"]
    store.insert_text_chunks.assert_not_called()
    store.delete_document.assert_called_once_with("doc-1")
    assert not pipeline.image_dir("doc-1").exists()


def test_stop_during_embedding_stores_nothing(settings, tmp_path):
    store, embedder = MagicMock(), FakeTextEmbedder()
    pipeline = make_pipeline(settings, FakeParser(), store, embedder)
    calls = 0

    def stop_after_first_embedding_batch() -> bool:
        nonlocal calls
        calls += 1
        return len(embedder.batches) >= 1

    with pytest.raises(IngestionCancelled):
        run(pipeline, tmp_path, should_stop=stop_after_first_embedding_batch)

    assert embedder.batches == [2]
    store.insert_text_chunks.assert_not_called()
    store.delete_document.assert_called_once_with("doc-1")


class SlowChunker(RecursiveChunker):
    """A chunker with two slow steps and one failed window, like the model-based strategies."""

    def split_texts(self, texts, checkpoint):
        for step in range(2):
            checkpoint("Chunking: sentences read by the chat model", step, 2)
        self.n_fallbacks = 1
        return super().split_texts(texts, checkpoint)


def slow_chunker_factory(seen: list):
    def build(config):
        seen.append(config)
        return SlowChunker(chunk_size=100, chunk_overlap=10, table_max_chars=500)

    return build


def test_chosen_chunking_reaches_the_factory_the_chunks_and_the_result(settings, tmp_path):
    store, seen, events = MagicMock(), [], []
    pipeline = make_pipeline(
        settings, FakeParser(), store, chunker_factory=slow_chunker_factory(seen)
    )
    chunking = LLMChunking(window_chars=3000)

    result = run(pipeline, tmp_path, chunking=chunking, on_progress=events.append)

    assert seen == [chunking]
    assert result.chunking == chunking and result.n_chunk_fallbacks == 1
    text_chunks, _vectors = store.insert_text_chunks.call_args.args
    assert {chunk.metadata["chunker"] for chunk in text_chunks} == {"llm"}
    chunk_events = [(e.message, e.current, e.total) for e in events if e.stage == "chunk"]
    assert chunk_events == [
        ("Chunking", None, None),
        ("Chunking: sentences read by the chat model", 0, 2),
        ("Chunking: sentences read by the chat model", 1, 2),
    ]


def test_stop_during_chunking_cancels_and_cleans_up(settings, tmp_path):
    store, events = MagicMock(), []
    pipeline = make_pipeline(
        settings, FakeParser(), store, chunker_factory=slow_chunker_factory([])
    )

    def stop_at_second_chunking_step() -> bool:
        return bool(events) and events[-1].current == 0 and events[-1].stage == "chunk"

    with pytest.raises(IngestionCancelled):
        run(pipeline, tmp_path, on_progress=events.append, should_stop=stop_at_second_chunking_step)

    store.insert_text_chunks.assert_not_called()
    store.delete_document.assert_called_once_with("doc-1")
    assert not pipeline.image_dir("doc-1").exists()


def test_failure_removes_vectors_and_images_then_reraises(settings, tmp_path):
    store = MagicMock()
    pipeline = make_pipeline(settings, FakeParser(error=RuntimeError("parser crashed")), store)

    with pytest.raises(RuntimeError, match="parser crashed"):
        run(pipeline, tmp_path)

    store.delete_document.assert_called_once_with("doc-1")
    assert not pipeline.image_dir("doc-1").exists()


def test_interruption_that_is_not_an_exception_still_cleans_up(settings, tmp_path):
    """Streamlit stops a script with a BaseException raised from the progress callback."""

    class ScriptStopped(BaseException):
        pass

    def stopped(progress: IngestionProgress) -> None:
        if progress.stage == "chunk":
            raise ScriptStopped

    store = MagicMock()
    pipeline = make_pipeline(settings, FakeParser(), store)
    with pytest.raises(ScriptStopped):
        run(pipeline, tmp_path, on_progress=stopped)

    store.delete_document.assert_called_once_with("doc-1")
    assert not pipeline.image_dir("doc-1").exists()


def test_document_with_no_content_fails_and_suggests_ocr_only_when_it_was_off(settings, tmp_path):
    pipeline = make_pipeline(settings, FakeParser(elements=[]))
    with pytest.raises(IngestionError, match="OCR option switched on"):
        run(pipeline, tmp_path)
    with pytest.raises(IngestionError) as error:
        run(pipeline, tmp_path, ocr=True)
    assert "OCR" not in str(error.value)


def test_text_only_document_does_not_load_the_image_model(settings, tmp_path):
    def no_image_model():
        raise AssertionError("image embedder should not be built")

    pipeline = make_pipeline(
        settings, FakeParser(elements=[ParsedElement(type=ChunkType.TEXT, content="hello")])
    )
    pipeline._image_embedder_factory = no_image_model
    assert pipeline.run(tmp_path / "t.txt", doc_id="d", filename="t.txt").n_image_chunks == 0


# --- parser --------------------------------------------------------------------------------


class FakeConverter:
    """Records the page ranges it is asked for and returns an empty document for each."""

    def __init__(self):
        self.ranges: list[tuple[int, int]] = []

    def convert(self, path, page_range=None):
        self.ranges.append(page_range)
        result = MagicMock()
        result.document.iterate_items.return_value = []
        return result


@pytest.fixture
def batching_parser(monkeypatch):
    def build(n_pages: int, **settings):
        parser = DoclingParser(Settings(_env_file=None, **settings))
        converter = FakeConverter()
        ocr_modes = []
        monkeypatch.setattr(
            parser, "_get_converter", lambda ocr: ocr_modes.append(ocr) or converter
        )
        monkeypatch.setattr("ragdoc.ingestion.parser.pdf_page_count", lambda path: n_pages)
        return parser, converter, ocr_modes

    return build


def test_pdf_is_parsed_in_page_batches(batching_parser, tmp_path):
    parser, converter, ocr_modes = batching_parser(9, parse_page_batch=4)
    seen = []

    parser.parse(tmp_path / "a.pdf", tmp_path / "img", ocr=True, on_pages=lambda *a: seen.append(a))

    assert converter.ranges == [(1, 4), (5, 8), (9, 9)]
    assert seen == [(1, 4, 9), (5, 8, 9), (9, 9, 9)]
    assert ocr_modes == [True]


def test_raising_from_the_page_callback_stops_before_that_batch(batching_parser, tmp_path):
    parser, converter, _ = batching_parser(12, parse_page_batch=4)

    def stop_at_second_batch(first, last, total):
        if first == 5:
            raise IngestionCancelled

    with pytest.raises(IngestionCancelled):
        parser.parse(tmp_path / "a.pdf", tmp_path / "img", on_pages=stop_at_second_batch)
    assert converter.ranges == [(1, 4)]


def test_time_limit_is_enforced_between_batches(batching_parser, tmp_path, monkeypatch):
    parser, converter, _ = batching_parser(12, parse_page_batch=4, parse_timeout_s=100)
    clock = iter([0.0, 10.0, 150.0])  # deadline set at 0; first batch at 10 s; second at 150 s
    monkeypatch.setattr("ragdoc.ingestion.parser.time.monotonic", lambda: next(clock))

    with pytest.raises(ParseTimeoutError, match="stopped before page 5 of 12"):
        parser.parse(tmp_path / "a.pdf", tmp_path / "img")
    assert converter.ranges == [(1, 4)]


def test_parser_reads_plain_text_without_docling(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello world", encoding="utf-8")
    [element] = DoclingParser(Settings(_env_file=None)).parse(path, tmp_path / "images")
    assert (element.type, element.content) == (ChunkType.TEXT, "hello world")


def test_parser_rejects_unsupported_extension(tmp_path):
    with pytest.raises(UnsupportedFileTypeError):
        DoclingParser(Settings(_env_file=None)).parse(tmp_path / "a.exe", tmp_path / "images")
