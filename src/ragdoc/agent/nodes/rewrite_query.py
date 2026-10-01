"""Produce the search query: standalone on the first pass, reworded on a retry."""

from langchain_core.messages import HumanMessage, SystemMessage

from ragdoc.agent import prompts
from ragdoc.agent.context import history_messages, text_of, usage_of
from ragdoc.agent.state import AgentDeps, AgentState, Node, NodeResult


def clean_query(text: str, fallback: str) -> str:
    """The model's query without wrapping quotes or extra lines; `fallback` if it is empty."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    query = lines[0].strip("\"'` ") if lines else ""
    return query or fallback


def make_rewrite_query(deps: AgentDeps) -> Node:
    def rewrite_query(state: AgentState) -> NodeResult:
        question = state["question"]
        history = state.get("history", [])
        # This node is re-entered only after `grade` found the context not relevant.
        is_retry = state.get("relevant") is False

        if not is_retry and not history:
            return NodeResult(
                summary="No earlier conversation, so the question is used as the search query",
                update={"search_query": question, "tried_queries": [question]},
                payload={"search_query": question},
            )

        if is_retry:
            messages = [
                SystemMessage(prompts.REWRITE_RETRY_SYSTEM),
                *history_messages(history),
                HumanMessage(
                    prompts.REWRITE_RETRY_USER.format(
                        question=question,
                        tried_queries="\n".join(
                            f"- {query}" for query in state.get("tried_queries", [question])
                        ),
                        reason=state.get("grade_reason", "not relevant"),
                    )
                ),
            ]
        else:
            messages = [
                SystemMessage(prompts.REWRITE_SYSTEM),
                *history_messages(history),
                HumanMessage(question),
            ]

        reply = deps.control_model.invoke(messages)
        query = clean_query(text_of(reply), fallback=question)
        update = {"search_query": query, "tried_queries": [query], "usage": usage_of(reply)}
        if is_retry:
            retries = state.get("retries", 0) + 1
            update["retries"] = retries
            summary = f'Retry {retries}: new search query "{query}"'
        else:
            summary = f'Search query: "{query}"'
        return NodeResult(summary=summary, update=update, payload={"search_query": query})

    return rewrite_query
