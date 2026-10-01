"""The whole agent against the real models, Milvus and PostgreSQL."""

import pytest

from ragdoc.agent.graph import build_deps, build_graph
from ragdoc.retrieval.retriever import get_retriever
from ragdoc.schemas import ChatEventType
from ragdoc.services.chat_service import ChatService
from ragdoc.storage.postgres.models import DocumentStatus

pytestmark = pytest.mark.integration


@pytest.fixture
def chat(sample_pdf, service_and_store, scoped_sessions):
    documents, store, settings = service_and_store
    document = documents.ingest(sample_pdf.name, sample_pdf.read_bytes())
    assert document.status is DocumentStatus.READY, document.error
    deps = build_deps(
        settings, retriever=get_retriever(settings, store), session_scope=scoped_sessions
    )
    return ChatService(settings, graph=build_graph(deps), session_scope=scoped_sessions)


def test_grounded_answer_is_streamed_cited_and_stored(chat):
    conversation = chat.create_conversation()

    events = list(chat.stream(conversation.id, "Who is the chief executive of Northwind Robotics?"))

    final = events[-1]
    assert final.type is ChatEventType.FINAL, final.text
    result = final.result
    assert "Okafor" in result.answer
    assert result.used_chunk_ids, "the answer should cite at least one source"
    assert any(event.type is ChatEventType.TOKEN for event in events)
    nodes = [event.trace.node for event in events if event.type is ChatEventType.TRACE]
    assert nodes[:4] == ["route", "rewrite_query", "retrieve", "grade"]
    assert nodes[-2:] == ["generate", "persist"]
    assert result.metrics["prompt_tokens"] > 0

    user, assistant = chat.get_messages(conversation.id)
    assert (user.role, user.content) == (
        "user",
        "Who is the chief executive of Northwind Robotics?",
    )
    assert assistant.content == result.answer
    assert str(assistant.id) == result.message_id
    assert [c.id for c in assistant.chunks] == [c.id for c in result.retrieval.chunks]
    assert assistant.used_chunk_ids == result.used_chunk_ids
    assert assistant.chunks[0].metadata["filename"] == "northwind_report.pdf"
    assert [event.node for event in assistant.trace][-1] == "generate"
    assert assistant.metrics["chat_model"]
    assert chat.list_conversations()[0].title.startswith("Who is the chief executive")

    # A follow-up that only makes sense with the history.
    follow_up = chat.ask(conversation.id, "In which city was that company founded?")
    assert "Rotterdam" in follow_up.answer
    assert len(chat.get_messages(conversation.id)) == 4


def test_greeting_is_answered_without_searching(chat):
    conversation = chat.create_conversation()
    result = chat.ask(conversation.id, "Hello!")
    assert result.retrieval is None
    assert [event.node for event in result.trace] == ["route", "direct_answer", "persist"]
    assert result.answer


def test_question_outside_the_documents_is_not_answered_from_them(chat):
    conversation = chat.create_conversation()
    result = chat.ask(conversation.id, "What is the boiling point of mercury in kelvin?")
    assert result.used_chunk_ids == []
    assert "629" not in result.answer and "630" not in result.answer
    assert result.trace[-2].payload["cited_sources"] == []
