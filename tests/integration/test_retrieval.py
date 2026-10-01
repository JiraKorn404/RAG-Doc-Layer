"""Retrieval over a really ingested document, with the real embedding models."""

import pytest

from ragdoc.retrieval.retriever import get_retriever
from ragdoc.schemas import ChunkType
from ragdoc.storage.postgres.models import DocumentStatus

pytestmark = pytest.mark.integration


@pytest.fixture
def ingested(sample_pdf, service_and_store):
    service, store, settings = service_and_store
    document = service.ingest(sample_pdf.name, sample_pdf.read_bytes())
    assert document.status is DocumentStatus.READY, document.error
    return document, get_retriever(settings, store), settings


def test_retrieves_text_tables_and_images_with_scores(ingested):
    document, retriever, settings = ingested

    result = retriever.retrieve("Who is the chief executive of Northwind Robotics?")

    assert "Mira Okafor" in result.text[0].content
    scores = [chunk.similarity_score for chunk in result.text]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= chunk.similarity_score <= 1.0 for chunk in result.chunks)
    assert all(chunk.rerank_score is None for chunk in result.chunks)
    assert all(chunk.doc_id == str(document.id) for chunk in result.chunks)
    assert all(chunk.metadata["filename"] == document.filename for chunk in result.chunks)
    assert result.timings["total_ms"] > 0

    result = retriever.retrieve("What was the operating profit in 2023?")
    tables = [chunk for chunk in result.text if chunk.chunk_type is ChunkType.TABLE]
    assert tables and "3.2" in tables[0].content

    result = retriever.retrieve("bar chart of yearly revenue")
    [image] = result.images
    assert image.chunk_type is ChunkType.IMAGE
    assert image.page == 2
    assert "Figure 1" in image.content
    assert settings.resolve_data_path(image.image_path).is_file()


def test_top_k_limits_and_document_filter(ingested):
    document, retriever, _settings = ingested

    result = retriever.retrieve("revenue", text_top_k=1, image_top_k=0)
    assert len(result.text) == 1 and result.images == []

    other = retriever.retrieve("revenue", doc_ids=["00000000-0000-0000-0000-000000000000"])
    assert other.is_empty

    same = retriever.retrieve("revenue", doc_ids=[str(document.id)])
    assert same.text and same.images
