"""Vector retrieval over the text and image collections."""

import time
from collections.abc import Callable, Sequence
from typing import Protocol

from langchain_core.embeddings import Embeddings

from ragdoc.config import Settings, get_settings
from ragdoc.llm.embeddings import get_text_embedder
from ragdoc.llm.image_embeddings import ImageEmbedder, get_image_embedder
from ragdoc.schemas import RetrievalResult
from ragdoc.storage.milvus_store import MilvusStore, get_milvus_store


class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        text_top_k: int | None = None,
        image_top_k: int | None = None,
        doc_ids: Sequence[str] | None = None,
    ) -> RetrievalResult:
        """Find the chunks most similar to `query`. `None` for a top-k means the configured
        default; `doc_ids` restricts the search to those documents."""
        ...


class VectorRetriever:
    """Embeds the query once per collection and takes the top-k from each.

    The text collection is searched with the text embedding of the query, the image collection
    with its CLIP text embedding. No reranking: the reranking phase will add a step after this.
    """

    def __init__(
        self,
        *,
        store: MilvusStore,
        text_embedder: Embeddings,
        image_embedder_factory: Callable[[], ImageEmbedder],
        settings: Settings | None = None,
    ):
        self._store = store
        self._text_embedder = text_embedder
        # A factory, so the image model is only loaded when images are actually searched.
        self._image_embedder_factory = image_embedder_factory
        self._settings = settings or get_settings()

    def retrieve(
        self,
        query: str,
        *,
        text_top_k: int | None = None,
        image_top_k: int | None = None,
        doc_ids: Sequence[str] | None = None,
    ) -> RetrievalResult:
        query = query.strip()
        if not query:
            return RetrievalResult(query=query)
        text_top_k = self._settings.text_top_k if text_top_k is None else text_top_k
        image_top_k = self._settings.image_top_k if image_top_k is None else image_top_k

        embed_s = search_s = 0.0
        started = time.perf_counter()

        text_hits = []
        if text_top_k > 0:
            mark = time.perf_counter()
            vector = self._text_embedder.embed_query(query)
            embed_s += time.perf_counter() - mark
            mark = time.perf_counter()
            text_hits = self._store.search_text(vector, text_top_k, doc_ids)
            search_s += time.perf_counter() - mark

        image_hits = []
        if image_top_k > 0:
            mark = time.perf_counter()
            vector = self._image_embedder_factory().embed_query(query)
            embed_s += time.perf_counter() - mark
            mark = time.perf_counter()
            image_hits = self._store.search_images(vector, image_top_k, doc_ids)
            search_s += time.perf_counter() - mark

        return RetrievalResult(
            query=query,
            text=text_hits,
            images=image_hits,
            timings={
                "embed_ms": embed_s * 1000,
                "search_ms": search_s * 1000,
                "total_ms": (time.perf_counter() - started) * 1000,
            },
        )


def get_retriever(settings: Settings | None = None, store: MilvusStore | None = None) -> Retriever:
    settings = settings or get_settings()
    return VectorRetriever(
        store=store or get_milvus_store(),
        text_embedder=get_text_embedder(settings),
        image_embedder_factory=lambda: get_image_embedder(settings),
        settings=settings,
    )
