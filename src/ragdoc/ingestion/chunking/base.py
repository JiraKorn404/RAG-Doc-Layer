"""The chunker interface."""

from collections.abc import Sequence
from typing import Any, Protocol

from ragdoc.ingestion.parser import ParsedElement
from ragdoc.schemas import Chunk


class Chunker(Protocol):
    def chunk(
        self,
        elements: Sequence[ParsedElement],
        *,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Turn parsed elements into chunks. `metadata` is copied onto every chunk."""
        ...
