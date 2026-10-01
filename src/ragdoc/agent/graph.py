"""The query graph.

route ─┬─> direct_answer ─────────────────────────────> persist
       └─> rewrite_query -> retrieve -> grade ─┬─> generate -> persist
                 ^                              │
                 └── (not relevant, retries left)
"""

import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from ragdoc.agent.nodes.direct_answer import make_direct_answer
from ragdoc.agent.nodes.generate import make_generate
from ragdoc.agent.nodes.grade import make_after_grade, make_grade
from ragdoc.agent.nodes.persist import make_persist
from ragdoc.agent.nodes.retrieve import make_retrieve
from ragdoc.agent.nodes.rewrite_query import make_rewrite_query
from ragdoc.agent.nodes.route import make_route
from ragdoc.agent.state import AgentDeps, AgentState, Node
from ragdoc.config import Settings, get_settings
from ragdoc.llm.chat import get_chat_model, get_vision_model
from ragdoc.retrieval.retriever import Retriever, get_retriever
from ragdoc.schemas import TraceEvent
from ragdoc.storage.postgres.session import session_scope as default_session_scope

# Nodes whose model output is the user-facing answer (their tokens are streamed to the UI).
ANSWER_NODES = ("generate", "direct_answer")


def traced(name: str, node: Node) -> Callable[[AgentState], dict[str, Any]]:
    """Wrap a node so it announces its start, is timed, and adds a `TraceEvent` to the state."""

    def run(state: AgentState) -> dict[str, Any]:
        try:
            get_stream_writer()({"node_start": name})
        except RuntimeError:
            pass  # called outside a graph run, e.g. in a unit test
        started_at = datetime.now(UTC)
        started = time.perf_counter()
        result = node(state)
        event = TraceEvent(
            node=name,
            summary=result.summary,
            payload=result.payload,
            started_at=started_at,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        return {**result.update, "trace": [event]}

    return run


def build_graph(deps: AgentDeps):
    builder = StateGraph(AgentState)
    nodes: dict[str, Node] = {
        "route": make_route(deps),
        "direct_answer": make_direct_answer(deps),
        "rewrite_query": make_rewrite_query(deps),
        "retrieve": make_retrieve(deps),
        "grade": make_grade(deps),
        "generate": make_generate(deps),
        "persist": make_persist(deps),
    }
    for name, node in nodes.items():
        builder.add_node(name, traced(name, node))

    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route",
        lambda state: state["route"],
        {"retrieve": "rewrite_query", "direct": "direct_answer"},
    )
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade",
        make_after_grade(deps),
        {"generate": "generate", "rewrite_query": "rewrite_query"},
    )
    builder.add_edge("generate", "persist")
    builder.add_edge("direct_answer", "persist")
    builder.add_edge("persist", END)
    return builder.compile()


def build_deps(
    settings: Settings | None = None,
    *,
    retriever: Retriever | None = None,
    session_scope=default_session_scope,
) -> AgentDeps:
    settings = settings or get_settings()
    return AgentDeps(
        settings=settings,
        control_model=get_chat_model(
            settings, reasoning=False, num_predict=settings.control_max_tokens
        ),
        grader_model=get_vision_model(
            settings, reasoning=False, num_predict=settings.control_max_tokens
        ),
        answer_model=get_vision_model(
            settings, reasoning=settings.chat_reasoning, num_predict=settings.answer_max_tokens
        ),
        retriever=retriever or get_retriever(settings),
        session_scope=session_scope,
    )
