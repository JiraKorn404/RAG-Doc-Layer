"""Semantic chunking: a new chunk starts where the topic changes between two sentences."""

from collections.abc import Sequence

from langchain_core.embeddings import Embeddings

from ragdoc.ingestion.chunking.base import BaseChunker, ChunkCheckpoint
from ragdoc.ingestion.chunking.sentences import split_sentences

PROGRESS_MESSAGE = "Chunking: embedding sentences"


class SemanticChunker(BaseChunker):
    def __init__(
        self,
        *,
        embedder: Embeddings,
        breakpoint_percentile: int,
        buffer_sentences: int,
        min_chunk_chars: int,
        max_chunk_chars: int,
        table_max_chars: int,
        batch_size: int,
    ):
        super().__init__(table_max_chars=table_max_chars)
        self._embedder = embedder
        self._percentile = breakpoint_percentile
        self._buffer = buffer_sentences
        self._min_chars = min_chunk_chars
        self._max_chars = max_chunk_chars
        self._batch_size = batch_size

    def split_texts(
        self, texts: Sequence[str], checkpoint: ChunkCheckpoint | None
    ) -> list[list[str]]:
        import numpy as np

        sentences = [split_sentences(text) for text in texts]

        # Each sentence is embedded together with its neighbours. A text of one sentence has
        # no boundary to find, so it is not embedded.
        groups: list[str] = []
        for units in sentences:
            if len(units) > 1:
                groups.extend(
                    "".join(units[max(0, i - self._buffer) : i + self._buffer + 1]).strip()
                    for i in range(len(units))
                )
        vectors: list[list[float]] = []
        for start in range(0, len(groups), self._batch_size):
            if checkpoint is not None:
                checkpoint(PROGRESS_MESSAGE, start, len(groups))
            vectors.extend(self._embedder.embed_documents(groups[start : start + self._batch_size]))

        # Cosine distance between each sentence and the next one, within each text.
        distances: list[list[float]] = []
        offset = 0
        for units in sentences:
            if len(units) < 2:
                distances.append([])
                continue
            matrix = np.asarray(vectors[offset : offset + len(units)], dtype=float)
            offset += len(units)
            norms = np.linalg.norm(matrix, axis=1)
            matrix = matrix / np.where(norms == 0, 1.0, norms)[:, None]
            distances.append((1.0 - np.sum(matrix[:-1] * matrix[1:], axis=1)).tolist())

        # One threshold for the whole document: what counts as a large change depends on how
        # much its sentences differ in general.
        every = [distance for per_text in distances for distance in per_text]
        threshold = float(np.percentile(every, self._percentile)) if every else 0.0

        result: list[list[str]] = []
        for units, per_text in zip(sentences, distances, strict=True):
            groups: list[list[str]] = []
            for index, unit in enumerate(units):
                starts_chunk = index == 0 or per_text[index - 1] > threshold
                # A group below the minimum is merged with the one after it.
                if groups and not (starts_chunk and self._size(groups[-1]) >= self._min_chars):
                    groups[-1].append(unit)
                else:
                    groups.append([unit])
            if len(groups) > 1 and self._size(groups[-1]) < self._min_chars:
                groups[-2:] = [groups[-2] + groups[-1]]
            result.append(self._limit_sentences(groups, self._max_chars))
        return result

    @staticmethod
    def _size(group: list[str]) -> int:
        return len("".join(group).strip())
