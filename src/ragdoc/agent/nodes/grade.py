"""Judge whether the retrieved context can answer the question, and where to go next."""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ragdoc.agent import prompts
from ragdoc.agent.context import invoke_decision, source_blocks
from ragdoc.agent.state import AgentDeps, AgentState, Node, NodeResult


class Grade(BaseModel):
    reason: str
    relevant: bool


def make_grade(deps: AgentDeps) -> Node:
    def grade(state: AgentState) -> NodeResult:
        retrieval = state["retrieval"]
        if retrieval.is_empty:
            reason = "Nothing was retrieved"
            return NodeResult(
                summary=f"Not relevant: {reason}",
                update={"relevant": False, "grade_reason": reason},
                payload={"relevant": False, "reason": reason},
            )

        messages = [
            SystemMessage(prompts.GRADE_SYSTEM),
            HumanMessage(
                [
                    *source_blocks(retrieval, deps.settings),
                    {
                        "type": "text",
                        "text": prompts.QUESTION_BLOCK.format(question=state["question"]),
                    },
                ]
            ),
        ]
        choice, reason, usage = invoke_decision(
            deps.grader_model, messages, label="Relevant", choices=("yes", "no")
        )
        if choice is None:
            # Let the answer step judge the sources rather than discard them on a parse failure.
            choice, reason = "yes", "Could not read the grade; using the sources"
        verdict = Grade(reason=reason, relevant=choice == "yes")

        label = "Relevant" if verdict.relevant else "Not relevant"
        return NodeResult(
            summary=f"{label}: {verdict.reason}",
            update={"relevant": verdict.relevant, "grade_reason": verdict.reason, "usage": usage},
            payload={"relevant": verdict.relevant, "reason": verdict.reason},
        )

    return grade


def make_after_grade(deps: AgentDeps):
    """The edge out of `grade`: answer, or rewrite the query and search again."""

    def after_grade(state: AgentState) -> Literal["generate", "rewrite_query"]:
        if state["relevant"]:
            return "generate"
        # An empty result means nothing is indexed; another query cannot change that.
        if state["retrieval"].is_empty:
            return "generate"
        if state.get("retries", 0) >= deps.settings.max_retries:
            return "generate"
        return "rewrite_query"

    return after_grade
