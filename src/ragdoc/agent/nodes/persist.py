"""Save the finished turn: both messages, the retrieved chunks, the trace and the metrics."""

import uuid
from datetime import UTC, datetime

from ragdoc.agent.state import AgentDeps, AgentState, Node, NodeResult
from ragdoc.schemas import TraceEvent
from ragdoc.storage.postgres.repositories import ConversationRepository, TraceRepository

DEFAULT_TITLE = "New conversation"
TITLE_CHARS = 60


def node_ms(trace: list[TraceEvent], *nodes: str) -> float:
    return sum(event.duration_ms for event in trace if event.node in nodes)


def build_metrics(state: AgentState, deps: AgentDeps) -> dict:
    trace = state.get("trace", [])
    usage = state.get("usage", {})
    started = trace[0].started_at if trace else datetime.now(UTC)
    return {
        "total_ms": (datetime.now(UTC) - started).total_seconds() * 1000,
        "retrieve_ms": node_ms(trace, "retrieve"),
        "grade_ms": node_ms(trace, "grade"),
        "generate_ms": node_ms(trace, "generate", "direct_answer"),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "n_retries": state.get("retries", 0),
        "chat_model": deps.settings.chat_model,
        "embed_model": deps.settings.text_embed_model,
    }


def make_persist(deps: AgentDeps) -> Node:
    def persist(state: AgentState) -> NodeResult:
        conversation_id = uuid.UUID(state["conversation_id"])
        question = state["question"]
        retrieval = state.get("retrieval")
        metrics = build_metrics(state, deps)

        # One transaction: a turn is saved completely or not at all.
        with deps.session_scope() as session:
            conversations = ConversationRepository(session)
            traces = TraceRepository(session)

            conversation = conversations.get(conversation_id)
            if conversation is None:
                raise LookupError(f"Conversation {conversation_id} not found")
            if conversation.title == DEFAULT_TITLE:
                title = " ".join(question.split())
                if len(title) > TITLE_CHARS:
                    title = title[: TITLE_CHARS - 3].rstrip() + "..."
                conversations.rename(conversation_id, title)

            conversations.add_message(conversation_id, role="user", content=question)
            message = conversations.add_message(
                conversation_id,
                role="assistant",
                content=state.get("answer", ""),
                reasoning=state.get("reasoning") or None,
            )
            if retrieval is not None:
                traces.add_retrieved_chunks(
                    message.id, retrieval.chunks, state.get("used_chunk_ids", [])
                )
            traces.add_trace_events(message.id, state.get("trace", []))
            traces.add_query_metrics(message.id, **metrics)
            message_id = str(message.id)

        return NodeResult(
            summary="Saved the question, answer, sources and trace",
            update={"message_id": message_id, "metrics": metrics},
        )

    return persist
