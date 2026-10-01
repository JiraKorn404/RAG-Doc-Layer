"""The retrieved chunks behind an answer, as cards."""

from collections.abc import Callable, Collection
from pathlib import Path

import streamlit as st

from components.text import safe_markdown
from ragdoc.schemas import ChunkType, RetrievedChunk

ImageLookup = Callable[[RetrievedChunk], Path | None]


def chunk_heading(number: int, chunk: RetrievedChunk, cited: bool) -> str:
    parts = [str(chunk.metadata.get("filename", "document"))]
    if chunk.page is not None:
        parts.append(f"page {chunk.page}")
    parts.append(chunk.chunk_type.value)
    badge = " :green-badge[cited in answer]" if cited else ""
    return f"**[{number}]** {' · '.join(parts)}{badge}"


def score_line(chunk: RetrievedChunk) -> str:
    line = f"Similarity score: {chunk.similarity_score:.4f}"
    if chunk.rerank_score is not None:
        line += f" · Rerank score: {chunk.rerank_score:.4f}"
    return line


def render_chunk(
    number: int, chunk: RetrievedChunk, cited: bool, image_lookup: ImageLookup
) -> None:
    with st.container(border=True):
        st.markdown(chunk_heading(number, chunk, cited))
        st.caption(score_line(chunk))
        if chunk.chunk_type is ChunkType.IMAGE:
            path = image_lookup(chunk)
            if path is not None:
                st.image(str(path), width=420)
            else:
                st.caption("The image file is no longer available (its document was deleted).")
            if chunk.content:
                st.caption(f"Caption: {chunk.content}")
        elif chunk.chunk_type is ChunkType.TABLE:
            # Stored as a Markdown table, so it renders as a table.
            st.markdown(safe_markdown(chunk.content))
        else:
            # Plain text: chunk text can contain headings that Markdown would blow up.
            st.text(chunk.content)


def render_sources(
    chunks: list[RetrievedChunk],
    used_chunk_ids: Collection[str],
    image_lookup: ImageLookup,
) -> None:
    """The collapsed sources section shown under a stored answer.

    A chunk's number is its position in `chunks`, which is the number the answer cites.
    """
    if not chunks:
        return
    numbered = list(enumerate(chunks, start=1))
    text = [(n, c) for n, c in numbered if c.chunk_type is not ChunkType.IMAGE]
    images = [(n, c) for n, c in numbered if c.chunk_type is ChunkType.IMAGE]
    n_cited = sum(chunk.id in used_chunk_ids for chunk in chunks)

    label = f"Retrieved chunks: {len(text)} text/table, {len(images)} image · {n_cited} cited"
    with st.expander(label):
        if text:
            st.markdown("**Text and tables**")
            for number, chunk in text:
                render_chunk(number, chunk, chunk.id in used_chunk_ids, image_lookup)
        if images:
            st.markdown("**Images**")
            for number, chunk in images:
                render_chunk(number, chunk, chunk.id in used_chunk_ids, image_lookup)
        if text and images:
            st.caption(
                "Text and image scores come from different embedding models, so they are not "
                "comparable with each other."
            )
