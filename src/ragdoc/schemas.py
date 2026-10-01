"""Pydantic models that cross layer boundaries."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, ClassVar, Literal, Self
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


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


class _Chunking(BaseModel):
    """How the text of one upload is split. Tables and images are handled the same way
    whatever the strategy.

    Each field's `title` and `description` are the label and help text of its input on the
    Documents tab, and its bounds are the input's range.
    """

    label: ClassVar[str]
    summary: ClassVar[str]

    @property
    def params(self) -> dict[str, int]:
        """The tunable values, without the strategy name."""
        return self.model_dump(exclude={"strategy"})


class RecursiveChunking(_Chunking):
    label: ClassVar[str] = "Recursive"
    summary: ClassVar[str] = (
        "Cuts the text into pieces of a fixed maximum size, at paragraph and sentence breaks "
        "where possible. Fast: no model is involved."
    )

    strategy: Literal["recursive"] = "recursive"
    chunk_size: int = Field(
        default=1000,
        ge=200,
        le=16000,
        title="Chunk size (characters)",
        description="Maximum number of characters in a chunk.",
    )
    chunk_overlap: int = Field(
        default=150,
        ge=0,
        le=8000,
        title="Overlap (characters)",
        description="Characters repeated between neighbouring chunks. Must be below the size.",
    )

    @model_validator(mode="after")
    def _overlap_below_size(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("The overlap must be smaller than the chunk size.")
        return self


class SemanticChunking(_Chunking):
    label: ClassVar[str] = "Semantic"
    summary: ClassVar[str] = (
        "Starts a new chunk where the topic changes, found by comparing the embeddings of "
        "neighbouring sentences. Slower: every sentence is embedded."
    )

    strategy: Literal["semantic"] = "semantic"
    breakpoint_percentile: int = Field(
        default=90,
        ge=50,
        le=99,
        title="Breakpoint percentile",
        description=(
            "A chunk boundary is placed where the change between neighbouring sentences is "
            "above this percentile of all changes in the document. Lower gives more, smaller "
            "chunks."
        ),
    )
    buffer_sentences: int = Field(
        default=1,
        ge=0,
        le=3,
        title="Buffer (sentences)",
        description=(
            "Sentences on each side that are joined to a sentence before it is embedded, so "
            "that very short sentences do not cause boundaries."
        ),
    )
    min_chunk_chars: int = Field(
        default=200,
        ge=0,
        le=2000,
        title="Minimum chunk size (characters)",
        description="A smaller chunk is merged into its neighbour.",
    )
    max_chunk_chars: int = Field(
        default=2000,
        ge=500,
        le=16000,
        title="Maximum chunk size (characters)",
        description="A larger chunk is cut into pieces of at most this size.",
    )

    @model_validator(mode="after")
    def _min_below_max(self) -> Self:
        if self.min_chunk_chars >= self.max_chunk_chars:
            raise ValueError("The minimum chunk size must be smaller than the maximum.")
        return self


class LLMChunking(_Chunking):
    label: ClassVar[str] = "LLM-based"
    summary: ClassVar[str] = (
        "The chat model reads the text and decides where each chunk starts. Slowest: about one "
        "model call per page."
    )

    strategy: Literal["llm"] = "llm"
    target_chunk_chars: int = Field(
        default=1000,
        ge=300,
        le=8000,
        title="Target chunk size (characters)",
        description="The size the model is asked to aim for. Text shorter than this is not sent.",
    )
    max_chunk_chars: int = Field(
        default=2000,
        ge=500,
        le=16000,
        title="Maximum chunk size (characters)",
        description="A larger chunk is cut into pieces of at most this size.",
    )
    window_chars: int = Field(
        default=6000,
        ge=2000,
        le=16000,
        title="Window (characters)",
        description=(
            "How much text the model sees per call. Larger means fewer calls with longer prompts."
        ),
    )

    @model_validator(mode="after")
    def _target_within_max(self) -> Self:
        if self.target_chunk_chars > self.max_chunk_chars:
            raise ValueError("The target chunk size must not exceed the maximum.")
        return self


ChunkingConfig = Annotated[
    RecursiveChunking | SemanticChunking | LLMChunking, Field(discriminator="strategy")
]
CHUNKING_MODELS: dict[str, type[_Chunking]] = {
    "recursive": RecursiveChunking,
    "semantic": SemanticChunking,
    "llm": LLMChunking,
}


def parse_chunking(strategy: str, params: dict[str, Any] | None = None) -> ChunkingConfig:
    """Build and validate a chunking configuration from a strategy name and its parameters."""
    try:
        model = CHUNKING_MODELS[strategy]
    except KeyError:
        raise ValueError(
            f"Unknown chunker {strategy!r}. Available: {', '.join(CHUNKING_MODELS)}"
        ) from None
    return model(**(params or {}))


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
