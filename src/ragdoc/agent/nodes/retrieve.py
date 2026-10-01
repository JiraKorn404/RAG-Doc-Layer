"""Search both collections for the current query."""

from ragdoc.agent.state import AgentDeps, AgentState, Node, NodeResult
from ragdoc.schemas import RetrievedChunk


def _brief(chunk: RetrievedChunk) -> dict:
    return {
        "id": chunk.id,
        "type": chunk.chunk_type.value,
        "filename": chunk.metadata.get("filename"),
        "page": chunk.page,
        "similarity_score": round(chunk.similarity_score, 4),
    }


def make_retrieve(deps: AgentDeps) -> Node:
    def retrieve(state: AgentState) -> NodeResult:
        query = state.get("search_query") or state["question"]
        result = deps.retriever.retrieve(query)

        if result.is_empty:
            summary = "Nothing found: no documents have been indexed"
        else:
            parts = [f"{len(result.text)} text/table chunks"]
            if result.text:
                parts[-1] += f" (best similarity {result.text[0].similarity_score:.2f})"
            parts.append(f"{len(result.images)} images")
            if result.images:
                parts[-1] += f" (best similarity {result.images[0].similarity_score:.2f})"
            summary = "Found " + " and ".join(parts)

        return NodeResult(
            summary=summary,
            update={"retrieval": result},
            payload={
                "search_query": query,
                "text": [_brief(chunk) for chunk in result.text],
                "images": [_brief(chunk) for chunk in result.images],
                "timings_ms": {key: round(value) for key, value in result.timings.items()},
            },
        )

    return retrieve
