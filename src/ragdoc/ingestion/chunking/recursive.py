"""Recursive character chunking for text; tables kept whole; one chunk per image."""

from collections.abc import Sequence
from typing import Any

from ragdoc.ingestion.parser import ParsedElement
from ragdoc.schemas import Chunk, ChunkType


def split_table(markdown: str, max_chars: int) -> list[str]:
    """Split a Markdown table by rows so each part fits `max_chars`, repeating the header.

    A table that already fits, or that has no recognisable header, is returned as is. A part
    always holds at least one row, so a single very long row can still exceed `max_chars`.
    """
    if len(markdown) <= max_chars:
        return [markdown]
    lines = markdown.splitlines()
    if len(lines) < 3 or not lines[1].lstrip().startswith("|") or "-" not in lines[1]:
        return [markdown]

    header = "\n".join(lines[:2])
    parts: list[str] = []
    rows: list[str] = []
    size = len(header)
    for row in lines[2:]:
        if rows and size + 1 + len(row) > max_chars:
            parts.append("\n".join([header, *rows]))
            rows, size = [], len(header)
        rows.append(row)
        size += 1 + len(row)
    if rows:
        parts.append("\n".join([header, *rows]))
    return parts


class RecursiveChunker:
    def __init__(self, *, chunk_size: int, chunk_overlap: int, table_max_chars: int):
        # Imported here: the langchain_text_splitters package pulls in torch on import, which
        # takes many seconds and is not needed just to list or delete documents.
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self._splitter_class = RecursiveCharacterTextSplitter
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self._table_max_chars = table_max_chars

    def chunk(
        self,
        elements: Sequence[ParsedElement],
        *,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []

        def add(chunk_type: ChunkType, content: str, element: ParsedElement) -> None:
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_type=chunk_type,
                    content=content,
                    page=element.page,
                    chunk_index=len(chunks),
                    image_path=str(element.image_path) if element.image_path else None,
                    metadata=dict(metadata or {}),
                )
            )

        for element in elements:
            if element.type is ChunkType.TEXT:
                for piece in self._splitter.split_text(element.content):
                    add(ChunkType.TEXT, piece, element)

            elif element.type is ChunkType.TABLE:
                for part in self._table_parts(element):
                    add(ChunkType.TABLE, part, element)

            elif element.type is ChunkType.IMAGE:
                add(ChunkType.IMAGE, element.caption, element)

        return chunks

    def _table_parts(self, element: ParsedElement) -> list[str]:
        prefix = f"{element.caption}\n\n" if element.caption else ""
        budget = max(self._table_max_chars - len(prefix), 1)
        parts: list[str] = []
        for part in split_table(element.content, budget):
            if len(part) <= budget:
                parts.append(prefix + part)
            else:
                # A row too long to fit even alone: fall back to plain splitting.
                splitter = self._splitter_class(
                    chunk_size=budget, chunk_overlap=0, separators=["\n", " | ", " ", ""]
                )
                parts.extend(splitter.split_text(part))
        return parts
