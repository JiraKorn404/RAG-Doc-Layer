"""Reply without searching (greetings, small talk, questions about the conversation)."""

from langchain_core.messages import HumanMessage, SystemMessage

from ragdoc.agent import prompts
from ragdoc.agent.context import history_messages, reasoning_of, text_of, usage_of
from ragdoc.agent.state import AgentDeps, AgentState, Node, NodeResult


def make_direct_answer(deps: AgentDeps) -> Node:
    def direct_answer(state: AgentState) -> NodeResult:
        reply = deps.answer_model.invoke(
            [
                SystemMessage(prompts.DIRECT_SYSTEM),
                *history_messages(state.get("history", [])),
                HumanMessage(state["question"]),
            ]
        )
        return NodeResult(
            summary="Answered without searching the documents",
            update={
                "answer": text_of(reply).strip(),
                "reasoning": reasoning_of(reply),
                "used_chunk_ids": [],
                "usage": usage_of(reply),
            },
        )

    return direct_answer
