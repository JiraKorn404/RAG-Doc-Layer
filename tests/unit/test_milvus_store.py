from unittest.mock import MagicMock

import pytest

from ragdoc.config import Settings
from ragdoc.schemas import Chunk, ChunkType
from ragdoc.storage.milvus_store import MilvusStore, doc_filter


@pytest.fixture
def client():
    return MagicMock()


@pytest.fixture
def store(client):
    return MilvusStore(Settings(_env_file=None), client=client)


def test_doc_filter_empty_means_no_restriction():
    assert doc_filter(None) == ""
    assert doc_filter([]) == ""


def test_doc_filter_quotes_and_escapes():
    assert doc_filter(["a", 'b"c']) == 'doc_id in ["a", "b\\"c"]'


def test_insert_nothing_does_not_call_milvus(store, client):
    assert store.insert_text_chunks([], []) == 0
    client.insert.assert_not_called()


def test_insert_rejects_mismatched_lengths(store):
    chunk = Chunk(doc_id="d", chunk_type=ChunkType.TEXT, content="x")
    with pytest.raises(ValueError):
        store.insert_text_chunks([chunk], [])


def test_search_with_zero_top_k_does_not_call_milvus(store, client):
    assert store.search_images([0.0], top_k=0) == []
    client.search.assert_not_called()


def test_search_text_maps_hits_to_retrieved_chunks(store, client):
    client.search.return_value = [
        [
            {
                "distance": 0.8,
                "entity": {
                    "id": "c1",
                    "doc_id": "d1",
                    "chunk_type": "table",
                    "content": "| a |",
                    "page": None,
                    "chunk_index": 3,
                    "metadata": {"filename": "f.pdf"},
                },
            }
        ]
    ]
    [hit] = store.search_text([0.0], top_k=1, doc_ids=["d1"])
    assert hit.chunk_type is ChunkType.TABLE
    assert hit.similarity_score == 0.8
    assert hit.page is None
    assert hit.metadata == {"filename": "f.pdf"}
    assert client.search.call_args.kwargs["filter"] == 'doc_id in ["d1"]'


def test_delete_document_hits_both_collections(store, client):
    store.delete_document("d1")
    collections = {call.kwargs["collection_name"] for call in client.delete.call_args_list}
    assert collections == {store.text_collection, store.image_collection}
