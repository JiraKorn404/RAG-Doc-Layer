"""Pydantic models that cross layer boundaries."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ChunkType(StrEnum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


class Chunk(BaseModel):
    """One retrievable unit of a document.

    For `text` and `table` chunks, `content` is the text (tables as Markdown). For `image`
    chunks, `content` is the caption (possibly empty) and `image_path` points at the file.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    doc_id: str
    chunk_type: ChunkType
    content: str = ""
    page: int | None = None
    chunk_index: int = 0
    image_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(Chunk):
    """A chunk returned by vector search, with its scores."""

    similarity_score: float
    # Filled by the reranking phase; always None until then.
    rerank_score: float | None = None


class RetrievalResult(BaseModel):
    """What retrieval found for one query.

    Text and image hits are separate lists, each best first. Their similarity scores come from
    different embedding models and are not comparable, so the lists must not be merged by score.
    """

    query: str
    text: list[RetrievedChunk] = Field(default_factory=list)
    images: list[RetrievedChunk] = Field(default_factory=list)
    # embed_ms, search_ms, total_ms
    timings: dict[str, float] = Field(default_factory=dict)

    @property
    def chunks(self) -> list[RetrievedChunk]:
        """All hits, text first then images. The order across the two groups is not a ranking."""
        return [*self.text, *self.images]

    @property
    def is_empty(self) -> bool:
        return not self.text and not self.images


class IngestionProgress(BaseModel):
    """Where an ingestion is, reported before each step and each batch."""

    stage: str  # wait, parse, chunk, embed or store
    message: str
    # Units already done and units in total (pages, chunks, images), when the step has them.
    current: int | None = None
    total: int | None = None
    elapsed_s: float = 0.0

    @property
    def fraction(self) -> float | None:
        """Share of the current step that is done, 0 to 1, or None if it is not countable."""
        if self.current is None or not self.total:
            return None
        return min(max(self.current / self.total, 0.0), 1.0)


class TraceEvent(BaseModel):
    """What one graph node did, for the UI trace and for persistence."""

    node: str
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    duration_ms: float = 0.0


class ChatResult(BaseModel):
    """One finished answer, with everything needed to show how it was produced."""

    conversation_id: str
    message_id: str
    answer: str
    reasoning: str = ""
    retrieval: RetrievalResult | None = None
    # Chunks the answer cites. A chunk's citation number is its 1-based position in
    # `retrieval.chunks`.
    used_chunk_ids: list[str] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class ChatEventType(StrEnum):
    NODE_START = "node_start"  # `node` is about to run
    TRACE = "trace"  # a node finished; `trace` describes what it did
    REASONING = "reasoning"  # a piece of the model's thinking, in `text`
    TOKEN = "token"  # a piece of the answer, in `text`
    FINAL = "final"  # the turn is saved; `result` holds everything
    ERROR = "error"  # the turn failed; `text` is a message for the user


class ChatEvent(BaseModel):
    """One item in the stream produced while answering a question."""

    type: ChatEventType
    node: str = ""
    text: str = ""
    trace: TraceEvent | None = None
    result: ChatResult | None = None
