"""The graph state and the pieces nodes share."""

import operator
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from sqlalchemy.orm import Session

from ragdoc.config import Settings
from ragdoc.retrieval.retriever import Retriever
from ragdoc.schemas import RetrievalResult, TraceEvent


def add_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    """Reducer: sum token counts from every model call in the turn."""
    return {key: left.get(key, 0) + right.get(key, 0) for key in left.keys() | right.keys()}


class HistoryMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class AgentState(TypedDict, total=False):
    # Input
    conversation_id: str
    question: str
    history: list[HistoryMessage]

    # Working state
    route: Literal["retrieve", "direct"]
    search_query: str
    tried_queries: Annotated[list[str], operator.add]
    retrieval: RetrievalResult
    relevant: bool
    grade_reason: str
    retries: int

    # Output
    answer: str
    reasoning: str
    used_chunk_ids: list[str]
    message_id: str
    metrics: dict[str, Any]

    # Accumulated across nodes
    trace: Annotated[list[TraceEvent], operator.add]
    usage: Annotated[dict[str, int], add_usage]


@dataclass
class NodeResult:
    """What a node returns: state updates, plus a line and details for the trace."""

    summary: str
    update: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)


Node = Callable[[AgentState], NodeResult]


@dataclass
class AgentDeps:
    """Everything nodes need from outside the graph."""

    settings: Settings
    # Short control decisions (routing, query rewriting): no thinking, text only.
    control_model: BaseChatModel
    # Judges retrieved context, which may include images: no thinking.
    grader_model: BaseChatModel
    # Writes the answer from text and images; thinks if CHAT_REASONING is on.
    answer_model: BaseChatModel
    retriever: Retriever
    session_scope: Callable[[], AbstractContextManager[Session]]
