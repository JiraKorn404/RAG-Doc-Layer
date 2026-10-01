"""Chunker factory. To add a strategy: implement `Chunker`, add its parameters model to
`schemas.py` (and its defaults to `Settings`), and register a builder here."""

from collections.abc import Callable

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from ragdoc.config import Settings, get_settings
from ragdoc.ingestion.chunking.base import Chunker
from ragdoc.ingestion.chunking.llm import LLMChunker
from ragdoc.ingestion.chunking.recursive import RecursiveChunker
from ragdoc.ingestion.chunking.semantic import SemanticChunker
from ragdoc.llm.chat import get_chunking_model
from ragdoc.llm.embeddings import get_text_embedder
from ragdoc.schemas import ChunkingConfig, LLMChunking, RecursiveChunking, SemanticChunking


def _recursive(config: RecursiveChunking, settings: Settings, **_models: object) -> Chunker:
    return RecursiveChunker(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        table_max_chars=settings.table_max_chars,
    )


def _semantic(
    config: SemanticChunking,
    settings: Settings,
    *,
    text_embedder: Embeddings | None = None,
    **_models: object,
) -> Chunker:
    return SemanticChunker(
        embedder=text_embedder or get_text_embedder(settings),
        breakpoint_percentile=config.breakpoint_percentile,
        buffer_sentences=config.buffer_sentences,
        min_chunk_chars=config.min_chunk_chars,
        max_chunk_chars=config.max_chunk_chars,
        table_max_chars=settings.table_max_chars,
        batch_size=settings.embed_batch_size,
    )


def _llm(
    config: LLMChunking,
    settings: Settings,
    *,
    chunking_model: BaseChatModel | None = None,
    **_models: object,
) -> Chunker:
    return LLMChunker(
        model=chunking_model or get_chunking_model(settings),
        target_chunk_chars=config.target_chunk_chars,
        max_chunk_chars=config.max_chunk_chars,
        window_chars=config.window_chars,
        table_max_chars=settings.table_max_chars,
    )


CHUNKERS: dict[str, Callable[..., Chunker]] = {
    "recursive": _recursive,
    "semantic": _semantic,
    "llm": _llm,
}


def get_chunker(
    config: ChunkingConfig | None = None,
    settings: Settings | None = None,
    *,
    text_embedder: Embeddings | None = None,
    chunking_model: BaseChatModel | None = None,
) -> Chunker:
    """The chunker for one upload. `config=None` means the configured default strategy.

    The models a strategy needs are built from `settings` unless passed in.
    """
    settings = settings or get_settings()
    config = config or settings.default_chunking()
    try:
        builder = CHUNKERS[config.strategy]
    except KeyError:
        raise ValueError(
            f"Unknown chunker {config.strategy!r}. Available: {', '.join(sorted(CHUNKERS))}"
        ) from None
    return builder(config, settings, text_embedder=text_embedder, chunking_model=chunking_model)
