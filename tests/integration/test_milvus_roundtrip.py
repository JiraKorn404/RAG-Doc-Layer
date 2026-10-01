import pytest

from ragdoc.schemas import Chunk, ChunkType

pytestmark = pytest.mark.integration

DIM = 8


def unit(axis: int) -> list[float]:
    vector = [0.0] * DIM
    vector[axis] = 1.0
    return vector


def test_text_vector_round_trip(milvus_store):
    chunks = [
        Chunk(
            doc_id="doc-a",
            chunk_type=ChunkType.TEXT,
            content="alpha",
            page=1,
            chunk_index=0,
            metadata={"filename": "a.pdf"},
        ),
        Chunk(doc_id="doc-a", chunk_type=ChunkType.TABLE, content="| x | y |", chunk_index=1),
        Chunk(doc_id="doc-b", chunk_type=ChunkType.TEXT, content="beta", page=7, chunk_index=0),
    ]
    assert milvus_store.insert_text_chunks(chunks, [unit(0), unit(1), unit(2)]) == 3

    hits = milvus_store.search_text(unit(0), top_k=3)
    assert [hit.content for hit in hits][0] == "alpha"
    best = hits[0]
    assert best.id == chunks[0].id
    assert best.similarity_score == pytest.approx(1.0, abs=1e-4)
    assert best.page == 1
    assert best.metadata == {"filename": "a.pdf"}
    assert best.rerank_score is None

    table = milvus_store.search_text(unit(1), top_k=1)[0]
    assert table.chunk_type is ChunkType.TABLE
    assert table.page is None


def test_search_can_be_restricted_to_documents(milvus_store):
    chunks = [
        Chunk(doc_id="doc-a", chunk_type=ChunkType.TEXT, content="alpha"),
        Chunk(doc_id="doc-b", chunk_type=ChunkType.TEXT, content="beta"),
    ]
    milvus_store.insert_text_chunks(chunks, [unit(0), unit(1)])

    hits = milvus_store.search_text(unit(0), top_k=5, doc_ids=["doc-b"])
    assert [hit.doc_id for hit in hits] == ["doc-b"]


def test_image_vector_round_trip(milvus_store):
    chunk = Chunk(
        doc_id="doc-a",
        chunk_type=ChunkType.IMAGE,
        content="Figure 1: revenue",
        page=2,
        image_path="data/images/doc-a/fig1.png",
    )
    assert milvus_store.insert_image_chunks([chunk], [unit(3)]) == 1

    [hit] = milvus_store.search_images(unit(3), top_k=3)
    assert hit.chunk_type is ChunkType.IMAGE
    assert hit.image_path == "data/images/doc-a/fig1.png"
    assert hit.content == "Figure 1: revenue"
    assert hit.page == 2


def test_delete_document_removes_it_from_both_collections(milvus_store):
    milvus_store.insert_text_chunks(
        [
            Chunk(doc_id="doc-a", chunk_type=ChunkType.TEXT, content="alpha"),
            Chunk(doc_id="doc-b", chunk_type=ChunkType.TEXT, content="beta"),
        ],
        [unit(0), unit(1)],
    )
    milvus_store.insert_image_chunks(
        [Chunk(doc_id="doc-a", chunk_type=ChunkType.IMAGE, image_path="a.png")], [unit(2)]
    )

    milvus_store.delete_document("doc-a")

    assert set(milvus_store.count_chunks("doc-a").values()) == {0}
    assert milvus_store.count_chunks("doc-b")[milvus_store.text_collection] == 1
    assert [hit.doc_id for hit in milvus_store.search_text(unit(0), top_k=5)] == ["doc-b"]
    assert milvus_store.search_images(unit(2), top_k=5) == []


def test_ensure_collections_is_idempotent(milvus_store):
    milvus_store.ensure_collections()
