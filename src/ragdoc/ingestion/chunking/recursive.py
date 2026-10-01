"""Recursive character chunking for text; tables kept whole; one chunk per image."""

from collections.abc import Sequence

from ragdoc.ingestion.chunking.base import BaseChunker, ChunkCheckpoint, split_table

__all__ = ["RecursiveChunker", "split_table"]


class RecursiveChunker(BaseChunker):
    def __init__(self, *, chunk_size: int, chunk_overlap: int, table_max_chars: int):
        super().__init__(table_max_chars=table_max_chars)
        self._splitter = self._splitter_class(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_texts(
        self, texts: Sequence[str], checkpoint: ChunkCheckpoint | None
    ) -> list[list[str]]:
        return [self._splitter.split_text(text) for text in texts]
