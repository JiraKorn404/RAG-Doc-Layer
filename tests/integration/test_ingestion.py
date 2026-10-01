"""End to end: a real PDF through Docling, the real embedding models, Milvus and PostgreSQL."""

import pytest

from ragdoc.llm.embeddings import get_text_embedder
from ragdoc.llm.image_embeddings import get_image_embedder
from ragdoc.schemas import ChunkType, LLMChunking, SemanticChunking
from ragdoc.services.document_service import DuplicateDocumentError
from ragdoc.storage.postgres.models import DocumentStatus

pytestmark = pytest.mark.integration


def test_ingest_search_and_delete(sample_pdf, service_and_store):
    service, store, settings = service_and_store
    stages = []

    document = service.ingest(sample_pdf.name, sample_pdf.read_bytes(), on_progress=stages.append)

    assert document.status is DocumentStatus.READY, document.error
    assert document.n_text_chunks >= 2
    assert document.n_table_chunks >= 1
    assert document.n_image_chunks >= 1
    assert stages[0].message == "Parsing document"
    assert stages[1].message == "Parsing pages 1-2 of 2"
    assert stages[-1].message == "Storing vectors"
    assert document.ocr_used is False and document.ingest_seconds > 0
    # The database may hold other documents: these tests run on the real one, rolled back.
    assert document.id in [d.id for d in service.list_documents()]

    doc_id = str(document.id)
    counts = store.count_chunks(doc_id)
    assert counts[store.text_collection] == document.n_text_chunks + document.n_table_chunks
    assert counts[store.image_collection] == document.n_image_chunks
    upload_dir = settings.uploads_dir / doc_id
    image_dir = settings.images_dir / doc_id
    assert (upload_dir / sample_pdf.name).is_file()
    assert len(list(image_dir.glob("*.png"))) == document.n_image_chunks

    # Text: the fact is on page 1.
    text_embedder = get_text_embedder(settings)
    hits = store.search_text(text_embedder.embed_query("Who is the chief executive?"), top_k=3)
    assert any("Mira Okafor" in hit.content for hit in hits)
    assert all(
        hit.metadata == {"filename": sample_pdf.name, "chunker": "recursive"} for hit in hits
    )

    # Table: stored as Markdown with its numbers intact.
    hits = store.search_text(
        text_embedder.embed_query("operating profit by year in a table"), top_k=5
    )
    tables = [hit for hit in hits if hit.chunk_type is ChunkType.TABLE]
    assert tables and "26.3" in tables[0].content and "|" in tables[0].content

    # Image: the chart is found with a text query, and its file exists.
    [image] = store.search_images(get_image_embedder(settings).embed_query("a bar chart"), top_k=1)
    assert image.page == 2
    assert settings.resolve_data_path(image.image_path).is_file()

    with pytest.raises(DuplicateDocumentError):
        service.ingest("renamed.pdf", sample_pdf.read_bytes())

    assert service.delete(document.id) is True
    assert set(store.count_chunks(doc_id).values()) == {0}
    assert not upload_dir.exists()
    assert not image_dir.exists()
    assert document.id not in [d.id for d in service.list_documents()]
    assert service.delete(document.id) is False


def test_failed_ingestion_is_listed_cleaned_up_and_can_be_retried(service_and_store):
    service, store, settings = service_and_store
    broken = b"%PDF-1.4 this is not really a pdf"

    failed = service.ingest("broken.pdf", broken)

    assert failed.status is DocumentStatus.FAILED
    assert failed.error
    assert set(store.count_chunks(str(failed.id)).values()) == {0}
    assert not (settings.images_dir / str(failed.id)).exists()

    retried = service.ingest("broken.pdf", broken)
    assert retried.id != failed.id
    listed = [d.id for d in service.list_documents()]
    assert retried.id in listed and failed.id not in listed


def test_plain_text_file(service_and_store):
    service, store, _settings = service_and_store
    document = service.ingest("notes.txt", b"The warehouse in Gdansk opened in 2024.")
    assert document.status is DocumentStatus.READY
    assert (document.n_text_chunks, document.n_table_chunks, document.n_image_chunks) == (1, 0, 0)
    assert service.delete(document.id) is True


TWO_TOPICS = (
    "The warehouse in Gdansk opened in March 2024. It stores spare parts for the conveyor "
    "systems. Forty people work there in two shifts. The building has twelve loading docks. "
    "Trucks arrive mostly in the early morning. A second warehouse is planned for Rotterdam.\n\n"
    "Sourdough bread needs a starter of flour and water. The starter is fed every day for a "
    "week. The dough rests overnight in the refrigerator. It is baked in a very hot oven with "
    "steam. A good loaf has a crisp crust and an open crumb. The bread keeps for several days."
)


@pytest.mark.parametrize(
    ("chunking", "progress_message"),
    [
        (
            SemanticChunking(breakpoint_percentile=80, min_chunk_chars=100, max_chunk_chars=600),
            "Chunking: embedding sentences",
        ),
        (
            LLMChunking(target_chunk_chars=300, max_chunk_chars=600, window_chars=2000),
            "Chunking: sentences read by the chat model",
        ),
    ],
)
def test_text_is_chunked_by_the_chosen_strategy(service_and_store, chunking, progress_message):
    """The real embedding and chat models decide the boundaries of a two-topic text."""
    service, store, settings = service_and_store
    progress = []

    document = service.ingest(
        "topics.txt", TWO_TOPICS.encode(), chunking=chunking, on_progress=progress.append
    )

    assert document.status is DocumentStatus.READY, document.error
    assert (document.chunker, document.chunk_params) == (chunking.strategy, chunking.params)
    assert document.chunk_fallbacks == 0
    assert progress_message in [p.message for p in progress if p.stage == "chunk"]
    assert document.n_text_chunks >= 2

    hits = store.search_text(get_text_embedder(settings).embed_query("sourdough bread"), top_k=20)
    assert len(hits) == document.n_text_chunks
    assert all(hit.metadata["chunker"] == chunking.strategy for hit in hits)
    assert all(len(hit.content) <= 600 for hit in hits)
    # The best hit is about bread and does not also contain the warehouse topic.
    assert "ourdough" in hits[0].content and "Gdansk" not in hits[0].content
    assert service.delete(document.id) is True


def test_multipage_pdf_is_parsed_in_batches_with_page_progress(tmp_path, service_and_store):
    """Real Docling, 9 pages in batches of 4: page numbers survive and nothing is lost."""
    import importlib.util

    from ragdoc.config import PROJECT_ROOT

    spec = importlib.util.spec_from_file_location(
        "make_sample_pdf", PROJECT_ROOT / "scripts" / "make_sample_pdf.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pdf = module.build_multipage_pdf(tmp_path / "logbook.pdf", pages=9)

    service, store, settings = service_and_store
    progress = []
    document = service.ingest(pdf.name, pdf.read_bytes(), on_progress=progress.append)

    assert document.status is DocumentStatus.READY, document.error
    parsing = [p.message for p in progress if p.stage == "parse"]
    assert parsing == [
        "Parsing document",
        "Parsing pages 1-4 of 9",
        "Parsing pages 5-8 of 9",
        "Parsing page 9 of 9",
    ]
    assert document.n_table_chunks == 3  # pages 3, 6, 9
    assert document.n_image_chunks == 2  # pages 4, 8
    images = sorted((settings.images_dir / str(document.id)).glob("*.png"))
    assert [image.name for image in images] == ["image_001.png", "image_002.png"]

    # A fact from the last batch is found, with its real page number.
    embedder = get_text_embedder(settings)
    hits = store.search_text(embedder.embed_query("Which unit has the marker word zebra9?"), 3)
    assert any("zebra9" in hit.content and hit.page == 9 for hit in hits)
    pages = {
        hit.page
        for hit in store.search_text(embedder.embed_query("quarterly notes logbook"), top_k=30)
    }
    assert pages == set(range(1, 10))
    assert service.delete(document.id) is True
