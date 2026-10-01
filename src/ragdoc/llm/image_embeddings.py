"""Image embedding model factory.

Ollama cannot embed images, so this runs a CLIP model in-process. Images and text queries are
embedded into the same space: `embed_images` for ingestion, `embed_query` for retrieval.
"""

import threading
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from ragdoc.config import Settings, get_settings
from ragdoc.llm.embeddings import check_dimension


class ImageEmbedder(Protocol):
    dim: int

    def embed_images(self, paths: Sequence[Path]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class ClipImageEmbedder:
    def __init__(self, model_name: str, dim: int):
        # Imported here so that importing ragdoc does not load torch.
        from sentence_transformers import SentenceTransformer

        self.model = model_name
        self.dim = dim
        self._model = SentenceTransformer(model_name)

    def embed_images(self, paths: Sequence[Path]) -> list[list[float]]:
        from PIL import Image

        if not paths:
            return []
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        vectors = self._model.encode(images, normalize_embeddings=True).tolist()
        check_dimension(vectors, self.dim, self.model)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode([text], normalize_embeddings=True)[0].tolist()
        check_dimension([vector], self.dim, self.model)
        return vector


_load_lock = threading.Lock()


@lru_cache(maxsize=1)
def _load(model_name: str, dim: int) -> ClipImageEmbedder:
    return ClipImageEmbedder(model_name, dim)


def get_image_embedder(settings: Settings | None = None) -> ImageEmbedder:
    """The image embedder, loaded once per process (loading the model takes seconds)."""
    settings = settings or get_settings()
    # The lock stops a warm-up thread and a first request from each loading the model.
    with _load_lock:
        return _load(settings.image_embed_model, settings.image_embed_dim)
