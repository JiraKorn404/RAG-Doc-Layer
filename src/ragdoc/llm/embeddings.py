"""Text embedding model factory."""

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from ragdoc.config import Settings, get_settings


class EmbeddingDimensionError(ValueError):
    """The embedding model returned vectors of a different size than configured."""


def check_dimension(vectors: list[list[float]], expected: int, model: str) -> None:
    for vector in vectors:
        if len(vector) != expected:
            raise EmbeddingDimensionError(
                f"{model} returned {len(vector)}-dim vectors but {expected} is configured. "
                "Fix the *_EMBED_DIM setting, then drop the collection and re-ingest."
            )


class TextEmbedder(Embeddings):
    """Wraps a text embedding model: instruction prefix on queries, dimension check on output."""

    def __init__(self, inner: Embeddings, *, dim: int, model: str, query_instruction: str = ""):
        self._inner = inner
        self.dim = dim
        self.model = model
        self._query_instruction = query_instruction

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._inner.embed_documents(texts)
        check_dimension(vectors, self.dim, self.model)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        if self._query_instruction:
            text = f"Instruct: {self._query_instruction}\nQuery: {text}"
        vector = self._inner.embed_query(text)
        check_dimension([vector], self.dim, self.model)
        return vector


def get_text_embedder(settings: Settings | None = None) -> TextEmbedder:
    settings = settings or get_settings()
    inner = OllamaEmbeddings(
        model=settings.text_embed_model,
        base_url=settings.ollama_base_url,
        keep_alive=settings.ollama_keep_alive_s,
        client_kwargs={"timeout": settings.ollama_timeout_s},
    )
    return TextEmbedder(
        inner,
        dim=settings.text_embed_dim,
        model=settings.text_embed_model,
        query_instruction=settings.text_embed_query_instruction,
    )
