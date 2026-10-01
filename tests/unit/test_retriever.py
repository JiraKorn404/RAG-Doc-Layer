from unittest.mock import MagicMock

import pytest

from ragdoc.config import Settings
from ragdoc.retrieval.retriever import VectorRetriever
from ragdoc.schemas import ChunkType, RetrievedChunk


def hit(chunk_id: str, chunk_type: ChunkType, score: float) -> RetrievedChunk:
    return RetrievedChunk(id=chunk_id, doc_id="d", chunk_type=chunk_type, similarity_score=score)


class FakeTextEmbedder:
    def embed_query(self, text):
        return [1.0, 0.0]


class FakeImageEmbedder:
    def embed_query(self, text):
        return [0.0, 1.0]


@pytest.fixture
def store():
    store = MagicMock()
    store.search_text.return_value = [
        hit("t1", ChunkType.TEXT, 0.8),
        hit("t2", ChunkType.TABLE, 0.6),
    ]
    store.search_images.return_value = [hit("i1", ChunkType.IMAGE, 0.27)]
    return store


def make_retriever(store, image_embedder_factory=FakeImageEmbedder, **settings):
    return VectorRetriever(
        store=store,
        text_embedder=FakeTextEmbedder(),
        image_embedder_factory=image_embedder_factory,
        settings=Settings(_env_file=None, **settings),
    )


def test_returns_separate_text_and_image_lists(store):
    result = make_retriever(store).retrieve("  what is x?  ")

    assert result.query == "what is x?"
    assert [c.id for c in result.text] == ["t1", "t2"]
    assert [c.id for c in result.images] == ["i1"]
    assert [c.id for c in result.chunks] == ["t1", "t2", "i1"]
    assert all(c.rerank_score is None for c in result.chunks)
    assert set(result.timings) == {"embed_ms", "search_ms", "total_ms"}


def test_each_collection_is_searched_with_its_own_embedding_and_default_top_k(store):
    make_retriever(store, text_top_k=7, image_top_k=2).retrieve("q")
    store.search_text.assert_called_once_with([1.0, 0.0], 7, None)
    store.search_images.assert_called_once_with([0.0, 1.0], 2, None)


def test_top_k_and_document_filter_can_be_overridden(store):
    make_retriever(store).retrieve("q", text_top_k=1, image_top_k=4, doc_ids=["a"])
    store.search_text.assert_called_once_with([1.0, 0.0], 1, ["a"])
    store.search_images.assert_called_once_with([0.0, 1.0], 4, ["a"])


def test_zero_image_top_k_skips_image_search_and_model(store):
    def no_image_model():
        raise AssertionError("image embedder should not be built")

    result = make_retriever(store, image_embedder_factory=no_image_model).retrieve(
        "q", image_top_k=0
    )
    assert result.images == []
    store.search_images.assert_not_called()


def test_zero_text_top_k_skips_text_search(store):
    result = make_retriever(store).retrieve("q", text_top_k=0)
    assert result.text == []
    store.search_text.assert_not_called()


def test_blank_query_returns_nothing_without_searching(store):
    result = make_retriever(store).retrieve("   ")
    assert result.is_empty
    store.search_text.assert_not_called()
    store.search_images.assert_not_called()
