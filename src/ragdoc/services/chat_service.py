"""Conversations and question answering. The UI's only entry point for chat."""

import logging
import uuid
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from langchain_core.messages import AIMessageChunk
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from ragdoc.agent.context import reasoning_of, text_of
from ragdoc.agent.graph import ANSWER_NODES, build_deps, build_graph
from ragdoc.config import Settings, get_settings
from ragdoc.llm.image_embeddings import get_image_embedder
from ragdoc.schemas import (
    ChatEvent,
    ChatEventType,
    ChatResult,
    ChunkType,
    RetrievedChunk,
    TraceEvent,
)
from ragdoc.storage.postgres.repositories import ConversationRepository, TraceRepository
from ragdoc.storage.postgres.session import session_scope

logger = logging.getLogger(__name__)

SessionScope = Callable[[], AbstractContextManager[Session]]


class ConversationInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime


class MessageView(BaseModel):
    """A stored message with, for assistant messages, how the answer was produced."""

    id: uuid.UUID
    role: str
    content: str
    reasoning: str = ""
    created_at: datetime
    # In citation order: the answer's [1] is the first chunk here.
    chunks: list[RetrievedChunk] = []
    used_chunk_ids: list[str] = []
    trace: list[TraceEvent] = []
    metrics: dict[str, Any] = {}


def describe_error(exc: BaseException, settings: Settings) -> str:
    """A message a user can act on, for the failures that are about the setup."""
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__

    for error in chain:
        module = type(error).__module__
        # Connected, but no data arrived in time: the server is there and too slow, which is a
        # different problem from not reaching it at all.
        if isinstance(error, httpx.ReadTimeout):
            return (
                f"Ollama at {settings.ollama_base_url} stopped responding (no data for "
                f"{settings.ollama_timeout_s:.0f} s). The Mac may be asleep, or still loading "
                "the model."
            )
        if isinstance(error, httpx.TransportError | ConnectionError):
            return (
                f"Could not reach Ollama at {settings.ollama_base_url}. Check that Tailscale is "
                "connected and that Ollama is running on the Mac."
            )
        if module.startswith("ollama"):
            return f"Ollama returned an error: {error}"
        if module.startswith("pymilvus"):
            return f"Milvus is not available ({error}). Is `docker compose up -d` running?"
        if module.startswith(("sqlalchemy", "psycopg")):
            return "PostgreSQL is not available. Is `docker compose up -d` running?"
    return f"{type(exc).__name__}: {exc}"


class ChatService:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        graph=None,
        session_scope: SessionScope = session_scope,
    ):
        self._settings = settings or get_settings()
        self._graph = graph
        self._session_scope = session_scope

    @property
    def graph(self):
        """Built on first use: it creates the model clients and the retriever."""
        if self._graph is None:
            self._graph = build_graph(build_deps(self._settings, session_scope=self._session_scope))
        return self._graph

    def warm_up(self) -> None:
        """Build the graph and load the image embedding model now, so the first question does
        not wait for them. Safe to call from a background thread."""
        _ = self.graph
        if self._settings.image_top_k > 0:
            get_image_embedder(self._settings)

    def image_file(self, chunk: RetrievedChunk) -> Path | None:
        """The file of an image chunk, or None if it has none or the file is gone (for example
        because its document was deleted)."""
        if not chunk.image_path:
            return None
        path = self._settings.resolve_data_path(chunk.image_path)
        return path if path.is_file() else None

    # --- conversations ------------------------------------------------------------------------

    def create_conversation(self, title: str | None = None) -> ConversationInfo:
        with self._session_scope() as session:
            return ConversationInfo.model_validate(ConversationRepository(session).create(title))

    def list_conversations(self) -> list[ConversationInfo]:
        """All conversations, most recently active first."""
        with self._session_scope() as session:
            return [
                ConversationInfo.model_validate(conversation)
                for conversation in ConversationRepository(session).list_all()
            ]

    def delete_conversation(self, conversation_id: uuid.UUID | str) -> bool:
        with self._session_scope() as session:
            return ConversationRepository(session).delete(uuid.UUID(str(conversation_id)))

    def get_messages(self, conversation_id: uuid.UUID | str) -> list[MessageView]:
        """The conversation in order, each answer with its sources, trace and metrics."""
        conversation_id = uuid.UUID(str(conversation_id))
        views: list[MessageView] = []
        with self._session_scope() as session:
            traces = TraceRepository(session)
            for message in ConversationRepository(session).list_messages(conversation_id):
                view = MessageView(
                    id=message.id,
                    role=message.role,
                    content=message.content,
                    reasoning=message.reasoning or "",
                    created_at=message.created_at,
                )
                if message.role == "assistant":
                    records = traces.list_retrieved_chunks(message.id)
                    view.chunks = [
                        RetrievedChunk(
                            id=record.chunk_id,
                            doc_id=record.doc_id,
                            chunk_type=ChunkType(record.chunk_type),
                            content=record.content_snapshot,
                            page=record.page,
                            image_path=record.image_path,
                            metadata=record.chunk_metadata or {},
                            similarity_score=record.similarity_score,
                            rerank_score=record.rerank_score,
                        )
                        for record in records
                    ]
                    view.used_chunk_ids = [r.chunk_id for r in records if r.used_in_answer]
                    view.trace = [
                        TraceEvent(
                            node=event.node,
                            summary=event.summary,
                            payload=event.payload or {},
                            started_at=event.started_at,
                            duration_ms=event.duration_ms,
                        )
                        for event in traces.list_trace_events(message.id)
                    ]
                    metrics = traces.get_query_metrics(message.id)
                    if metrics is not None:
                        view.metrics = {
                            column.name: getattr(metrics, column.name)
                            for column in metrics.__table__.columns
                            if column.name not in ("id", "message_id")
                        }
                views.append(view)
        return views

    # --- answering ----------------------------------------------------------------------------

    def stream(self, conversation_id: uuid.UUID | str, question: str) -> Iterator[ChatEvent]:
        """Answer a question, yielding events as the graph works.

        Order: NODE_START and TRACE around every step, REASONING and TOKEN pieces while the
        answer is written, then one FINAL carrying the saved result. On failure, one ERROR
        instead of FINAL; nothing is saved for a failed turn.
        """
        question = question.strip()
        if not question:
            yield ChatEvent(type=ChatEventType.ERROR, text="The question is empty.")
            return

        try:
            conversation_id = str(uuid.UUID(str(conversation_id)))
            state = {
                "conversation_id": conversation_id,
                "question": question,
                "history": self._history(conversation_id),
            }
            final: dict[str, Any] = {}
            modes = ["custom", "updates", "messages", "values"]
            for mode, data in self.graph.stream(state, stream_mode=modes):
                if mode == "custom":
                    if "node_start" in data:
                        yield ChatEvent(type=ChatEventType.NODE_START, node=data["node_start"])
                elif mode == "updates":
                    for node, update in data.items():
                        for event in (update or {}).get("trace", []):
                            yield ChatEvent(type=ChatEventType.TRACE, node=node, trace=event)
                elif mode == "messages":
                    chunk, metadata = data
                    node = metadata.get("langgraph_node", "")
                    # Only the answer is streamed; routing and grading output stays internal.
                    if node not in ANSWER_NODES or not isinstance(chunk, AIMessageChunk):
                        continue
                    if reasoning := reasoning_of(chunk):
                        yield ChatEvent(type=ChatEventType.REASONING, node=node, text=reasoning)
                    if text := text_of(chunk):
                        yield ChatEvent(type=ChatEventType.TOKEN, node=node, text=text)
                elif mode == "values":
                    final = data
        except Exception as exc:
            logger.exception("Answering failed for conversation %s", conversation_id)
            yield ChatEvent(type=ChatEventType.ERROR, text=describe_error(exc, self._settings))
            return

        yield ChatEvent(
            type=ChatEventType.FINAL,
            result=ChatResult(
                conversation_id=conversation_id,
                message_id=final.get("message_id", ""),
                answer=final.get("answer", ""),
                reasoning=final.get("reasoning", ""),
                retrieval=final.get("retrieval"),
                used_chunk_ids=final.get("used_chunk_ids", []),
                trace=final.get("trace", []),
                metrics=final.get("metrics", {}),
            ),
        )

    def ask(self, conversation_id: uuid.UUID | str, question: str) -> ChatResult:
        """Answer a question without streaming. Raises RuntimeError if the turn fails."""
        for event in self.stream(conversation_id, question):
            if event.type is ChatEventType.FINAL:
                return event.result
            if event.type is ChatEventType.ERROR:
                raise RuntimeError(event.text)
        raise RuntimeError("The answer stream ended without a result.")

    def _history(self, conversation_id: str) -> list[dict[str, str]]:
        """The last HISTORY_TURNS question/answer pairs, oldest first."""
        with self._session_scope() as session:
            conversations = ConversationRepository(session)
            if conversations.get(uuid.UUID(conversation_id)) is None:
                raise LookupError(f"Conversation {conversation_id} not found")
            messages = conversations.list_messages(
                uuid.UUID(conversation_id), limit=self._settings.history_turns * 2
            )
            return [{"role": m.role, "content": m.content} for m in messages]
