from ragdoc.schemas import Chunk, ChunkType, RetrievedChunk, TraceEvent


def test_chunk_gets_unique_id():
    first = Chunk(doc_id="d", chunk_type=ChunkType.TEXT, content="a")
    second = Chunk(doc_id="d", chunk_type=ChunkType.TEXT, content="a")
    assert first.id != second.id


def test_chunk_type_accepts_string_value():
    assert Chunk(doc_id="d", chunk_type="table").chunk_type is ChunkType.TABLE


def test_retrieved_chunk_has_no_rerank_score_by_default():
    chunk = RetrievedChunk(doc_id="d", chunk_type=ChunkType.IMAGE, similarity_score=0.3)
    assert chunk.rerank_score is None


def test_trace_event_timestamp_is_timezone_aware():
    assert TraceEvent(node="retrieve", summary="ok").started_at.tzinfo is not None
