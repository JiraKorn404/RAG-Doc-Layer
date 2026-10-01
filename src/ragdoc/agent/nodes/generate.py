"""Write the answer from the retrieved sources, with citations."""

from langchain_core.messages import HumanMessage, SystemMessage

from ragdoc.agent import prompts
from ragdoc.agent.context import (
    cited_chunk_ids,
    history_messages,
    reasoning_of,
    source_blocks,
    text_of,
    usage_of,
)
from ragdoc.agent.state import AgentDeps, AgentState, Node, NodeResult


def make_generate(deps: AgentDeps) -> Node:
    def generate(state: AgentState) -> NodeResult:
        question = state["question"]
        history = history_messages(state.get("history", []))
        retrieval = state["retrieval"]

        if retrieval.is_empty:
            # Nothing is indexed, so there is nothing to answer from.
            reply = deps.answer_model.invoke(
                [SystemMessage(prompts.NO_CONTEXT_SYSTEM), *history, HumanMessage(question)]
            )
            return NodeResult(
                summary="Told the user the documents do not cover this question",
                update={
                    "answer": text_of(reply).strip(),
                    "reasoning": reasoning_of(reply),
                    "used_chunk_ids": [],
                    "usage": usage_of(reply),
                },
                payload={"grounded": False},
            )

        # The sources are used even when `grade` judged them not relevant (which only drives
        # the retries). The grader is sometimes wrong, for example about what an image shows,
        # and the answer prompt already tells the model to say so when the sources do not
        # cover the question. A wrong "not relevant" then costs retries, not the answer.
        graded_relevant = bool(state.get("relevant"))
        reply = deps.answer_model.invoke(
            [
                SystemMessage(prompts.GENERATE_SYSTEM),
                *history,
                HumanMessage(
                    [
                        *source_blocks(retrieval, deps.settings),
                        {"type": "text", "text": prompts.QUESTION_BLOCK.format(question=question)},
                    ]
                ),
            ]
        )
        answer = text_of(reply).strip()
        used = cited_chunk_ids(answer, retrieval)
        cited = [n for n, chunk in enumerate(retrieval.chunks, start=1) if chunk.id in used]
        if cited:
            summary = "Answered, citing sources " + ", ".join(f"[{n}]" for n in cited)
        elif graded_relevant:
            summary = "Answered without citing a source"
        else:
            summary = "Found nothing in the sources that answers the question"
        return NodeResult(
            summary=summary,
            update={
                "answer": answer,
                "reasoning": reasoning_of(reply),
                "used_chunk_ids": used,
                "usage": usage_of(reply),
            },
            payload={
                "grounded": True,
                "graded_relevant": graded_relevant,
                "cited_sources": cited,
            },
        )

    return generate
