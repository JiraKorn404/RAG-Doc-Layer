import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from ragdoc.schemas import ChunkType, RetrievedChunk, TraceEvent
from ragdoc.storage.postgres.models import DocumentStatus
from ragdoc.storage.postgres.repositories import (
    ConversationRepository,
    DocumentRepository,
    TraceRepository,
)

pytestmark = pytest.mark.integration


def new_hash() -> str:
    return uuid.uuid4().hex + uuid.uuid4().hex


def test_document_lifecycle(db_session):
    documents = DocumentRepository(db_session)
    file_hash = new_hash()

    created = documents.create(filename="report.pdf", file_hash=file_hash, file_path="x/report.pdf")
    assert created.status == DocumentStatus.PROCESSING
    assert created.created_at is not None
    assert documents.get_by_hash(file_hash).id == created.id

    documents.mark_ready(created.id, n_text_chunks=10, n_table_chunks=2, n_image_chunks=3)
    documents.add_ingestion_metrics(
        created.id, parse_ms=1, chunk_ms=2, embed_ms=3, store_ms=4, total_ms=10
    )
    db_session.expire_all()

    loaded = documents.get(created.id)
    assert loaded.status == DocumentStatus.READY
    assert (loaded.n_text_chunks, loaded.n_table_chunks, loaded.n_image_chunks) == (10, 2, 3)
    assert created.id in [document.id for document in documents.list_all()]
    assert documents.get_ingestion_metrics(created.id).total_ms == 10

    assert documents.delete(created.id) is True
    db_session.expire_all()
    assert documents.get(created.id) is None
    assert documents.get_ingestion_metrics(created.id) is None
    assert documents.delete(created.id) is False


def test_document_failure_records_error(db_session):
    documents = DocumentRepository(db_session)
    created = documents.create(filename="bad.pdf", file_hash=new_hash(), file_path="x/bad.pdf")
    documents.mark_failed(created.id, "parser crashed")
    db_session.expire_all()
    loaded = documents.get(created.id)
    assert (loaded.status, loaded.error) == (DocumentStatus.FAILED, "parser crashed")


def test_duplicate_file_hash_is_rejected(db_session):
    documents = DocumentRepository(db_session)
    file_hash = new_hash()
    documents.create(filename="a.pdf", file_hash=file_hash, file_path="x/a.pdf")
    with pytest.raises(IntegrityError):
        documents.create(filename="b.pdf", file_hash=file_hash, file_path="x/b.pdf")


def test_messages_keep_their_order_and_limit(db_session):
    conversations = ConversationRepository(db_session)
    conversation = conversations.create("Revenue questions")
    for index in range(4):
        role = "user" if index % 2 == 0 else "assistant"
        conversations.add_message(conversation.id, role=role, content=f"m{index}")

    assert [m.content for m in conversations.list_messages(conversation.id)] == [
        "m0",
        "m1",
        "m2",
        "m3",
    ]
    assert [m.content for m in conversations.list_messages(conversation.id, limit=2)] == [
        "m2",
        "m3",
    ]


def test_answer_trace_round_trip_and_cascade(db_session):
    conversations = ConversationRepository(db_session)
    traces = TraceRepository(db_session)
    conversation = conversations.create()
    message = conversations.add_message(
        conversation.id, role="assistant", content="42", reasoning="because"
    )

    chunks = [
        RetrievedChunk(
            id="c1",
            doc_id="d1",
            chunk_type=ChunkType.TEXT,
            content="the answer is 42",
            page=3,
            similarity_score=0.81,
        ),
        RetrievedChunk(
            id="c2",
            doc_id="d1",
            chunk_type=ChunkType.IMAGE,
            content="",
            image_path="data/images/d1/fig.png",
            similarity_score=0.27,
        ),
    ]
    traces.add_retrieved_chunks(message.id, chunks, used_chunk_ids={"c1"})
    traces.add_trace_events(
        message.id,
        [
            TraceEvent(node="retrieve", summary="2 chunks", payload={"k": 5}, duration_ms=12.5),
            TraceEvent(node="generate", summary="answered", duration_ms=900),
        ],
    )
    traces.add_query_metrics(
        message.id, total_ms=950, retrieve_ms=12.5, generate_ms=900, chat_model="m", n_retries=1
    )
    db_session.expire_all()

    stored = traces.list_retrieved_chunks(message.id)
    assert [(c.chunk_id, c.rank, c.used_in_answer) for c in stored] == [
        ("c1", 1, True),
        ("c2", 2, False),
    ]
    assert stored[0].similarity_score == pytest.approx(0.81)
    assert stored[0].rerank_score is None
    assert stored[1].image_path == "data/images/d1/fig.png"

    events = traces.list_trace_events(message.id)
    assert [e.node for e in events] == ["retrieve", "generate"]
    assert events[0].payload == {"k": 5}
    assert traces.get_query_metrics(message.id).n_retries == 1

    # Deleting the conversation cascades to everything recorded under it.
    conversation_id, message_id = conversation.id, message.id
    assert conversations.delete(conversation_id) is True
    db_session.expire_all()
    assert conversations.list_messages(conversation_id) == []
    assert traces.list_retrieved_chunks(message_id) == []
    assert traces.list_trace_events(message_id) == []
    assert traces.get_query_metrics(message_id) is None


def test_add_message_to_missing_conversation_fails(db_session):
    with pytest.raises(LookupError):
        ConversationRepository(db_session).add_message(uuid.uuid4(), role="user", content="hi")
