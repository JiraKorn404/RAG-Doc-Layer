"""How an answer was produced: the steps the agent took, its thinking and the timings."""

from typing import Any

import streamlit as st

from components.text import safe_markdown, seconds
from ragdoc.schemas import TraceEvent

# What each graph node is called in the UI, while running and once done.
NODE_LABELS = {
    "route": ("Deciding whether to search the documents", "Route"),
    "direct_answer": ("Writing a reply", "Reply"),
    "rewrite_query": ("Writing the search query", "Search query"),
    "retrieve": ("Searching the documents", "Search"),
    "grade": ("Checking whether the results are relevant", "Relevance check"),
    "generate": ("Writing the answer", "Answer"),
    "persist": ("Saving", "Save"),
}


def running_label(node: str) -> str:
    return NODE_LABELS.get(node, (node, node))[0] + "..."


def step_line(event: TraceEvent) -> str:
    """One finished step as a Markdown line."""
    name = NODE_LABELS.get(event.node, (event.node, event.node))[1]
    return f"**{name}** · {seconds(event.duration_ms)}  \n{safe_markdown(event.summary)}"


def render_steps(trace: list[TraceEvent], *, show_data: bool = False) -> None:
    for event in trace:
        if event.node == "persist":
            continue
        st.markdown(step_line(event))
        if show_data and event.payload:
            st.json(event.payload, expanded=False)


def render_metrics(metrics: dict[str, Any]) -> None:
    if not metrics:
        return
    parts = [
        f"Total {seconds(metrics.get('total_ms', 0))}",
        f"search {seconds(metrics.get('retrieve_ms', 0))}",
        f"relevance check {seconds(metrics.get('grade_ms', 0))}",
        f"answer {seconds(metrics.get('generate_ms', 0))}",
        f"tokens in/out {metrics.get('prompt_tokens', 0)}/{metrics.get('completion_tokens', 0)}",
        f"search retries {metrics.get('n_retries', 0)}",
    ]
    if metrics.get("chat_model"):
        parts.append(str(metrics["chat_model"]))
    st.caption(" · ".join(parts))


def render_process(
    trace: list[TraceEvent], reasoning: str, metrics: dict[str, Any], *, key: str
) -> None:
    """The collapsed "how this was produced" section shown above a stored answer."""
    steps = [event for event in trace if event.node != "persist"]
    if not steps and not reasoning:
        return
    with st.expander(f"Thinking and steps ({len(steps)} steps)"):
        show_data = st.toggle("Show step data", key=f"step-data-{key}")
        render_steps(steps, show_data=show_data)
        if reasoning:
            st.markdown("**Model thinking**")
            st.markdown(safe_markdown(reasoning))
        render_metrics(metrics)
