"""Chunker factory. To add a chunker: implement `Chunker` and register a builder here."""

from collections.abc import Callable

from ragdoc.config import Settings, get_settings
from ragdoc.ingestion.chunking.base import Chunker
from ragdoc.ingestion.chunking.recursive import RecursiveChunker


def _recursive(settings: Settings) -> Chunker:
    return RecursiveChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        table_max_chars=settings.table_max_chars,
    )


CHUNKERS: dict[str, Callable[[Settings], Chunker]] = {
    "recursive": _recursive,
}


def get_chunker(settings: Settings | None = None) -> Chunker:
    settings = settings or get_settings()
    try:
        builder = CHUNKERS[settings.chunker]
    except KeyError:
        raise ValueError(
            f"Unknown chunker {settings.chunker!r}. Available: {', '.join(sorted(CHUNKERS))}"
        ) from None
    return builder(settings)
