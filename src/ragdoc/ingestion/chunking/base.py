"""The chunker interface, and what every strategy shares: tables, images, size limits."""

import math
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from ragdoc.ingestion.parser import ParsedElement
from ragdoc.schemas import Chunk, ChunkType

# Called by a chunker before each slow step (an embedding batch, a model call) with a message,
# the units done and the units in total. It reports progress, and raises to stop the ingestion.
ChunkCheckpoint = Callable[[str, int, int], None]


class Chunker(Protocol):
    # How many times the strategy could not be applied and plain splitting was used instead.
    n_fallbacks: int

    def chunk(
        self,
        elements: Sequence[ParsedElement],
        *,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
        checkpoint: ChunkCheckpoint | None = None,
    ) -> list[Chunk]:
        """Turn parsed elements into chunks. `metadata` is copied onto every chunk."""
        ...


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


class BaseChunker:
    """Tables kept whole, one chunk per image. A strategy only decides how text is split, by
    implementing `split_texts`."""

    def __init__(self, *, table_max_chars: int):
        # Imported here: the langchain_text_splitters package pulls in torch on import, which
        # takes many seconds and is not needed just to list or delete documents.
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self._splitter_class = RecursiveCharacterTextSplitter
        self._table_max_chars = table_max_chars
        self.n_fallbacks = 0

    def split_texts(
        self, texts: Sequence[str], checkpoint: ChunkCheckpoint | None
    ) -> list[list[str]]:
        """The pieces of each text, in order: one list per text. A text is one text element,
        which the parser limits to one page, so no piece spans pages."""
        raise NotImplementedError

    def chunk(
        self,
        elements: Sequence[ParsedElement],
        *,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
        checkpoint: ChunkCheckpoint | None = None,
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

        texts = [element.content for element in elements if element.type is ChunkType.TEXT]
        text_pieces = iter(self.split_texts(texts, checkpoint))

        for element in elements:
            if element.type is ChunkType.TEXT:
                for piece in next(text_pieces):
                    add(ChunkType.TEXT, piece, element)

            elif element.type is ChunkType.TABLE:
                for part in self._table_parts(element):
                    add(ChunkType.TABLE, part, element)

            elif element.type is ChunkType.IMAGE:
                add(ChunkType.IMAGE, element.caption, element)

        return chunks

    def _limit(self, pieces: Sequence[str], max_chars: int) -> list[str]:
        """Strip the pieces, drop empty ones and cut any over `max_chars` by plain splitting."""
        limited: list[str] = []
        for piece in pieces:
            piece = piece.strip()
            if len(piece) <= max_chars:
                if piece:
                    limited.append(piece)
            else:
                splitter = self._splitter_class(chunk_size=max_chars, chunk_overlap=0)
                limited.extend(splitter.split_text(piece))
        return limited

    def _limit_sentences(self, groups: Sequence[Sequence[str]], max_chars: int) -> list[str]:
        """One piece per group of sentences. A group over `max_chars` is cut between
        sentences, into parts of similar size; only a single sentence that is itself too long
        is cut inside."""
        pieces: list[str] = []
        for group in groups:
            total = len("".join(group).strip())
            if total <= max_chars:
                pieces.append("".join(group))
                continue
            target = total / math.ceil(total / max_chars)
            current = ""
            for sentence in group:
                too_long = len((current + sentence).strip()) > max_chars
                if current and (len(current) >= target or too_long):
                    pieces.append(current)
                    current = ""
                current += sentence
            pieces.append(current)
        return self._limit(pieces, max_chars)

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
