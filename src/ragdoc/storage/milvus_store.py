"""Milvus access: two collections, one for text/table chunks and one for image chunks."""

import json
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from pymilvus import DataType, MilvusClient

from ragdoc.config import Settings, get_settings
from ragdoc.schemas import Chunk, ChunkType, RetrievedChunk

METRIC = "COSINE"
INDEX_PARAMS = {"M": 16, "efConstruction": 200}
SEARCH_PARAMS = {"metric_type": METRIC, "params": {"ef": 64}}

# Milvus VARCHAR lengths are in bytes; 65535 is the maximum.
MAX_CONTENT_BYTES = 65535
MAX_CAPTION_BYTES = 8192

TEXT_FIELDS = ["id", "doc_id", "chunk_type", "content", "page", "chunk_index", "metadata"]
IMAGE_FIELDS = ["id", "doc_id", "image_path", "caption", "page", "metadata"]


def doc_filter(doc_ids: Sequence[str] | None) -> str:
    """Milvus filter expression restricting to the given documents (empty = no restriction)."""
    if not doc_ids:
        return ""
    # json.dumps gives a correctly quoted and escaped string-list literal.
    return f"doc_id in {json.dumps(list(doc_ids))}"


class MilvusStore:
    def __init__(self, settings: Settings | None = None, client: MilvusClient | None = None):
        self._settings = settings or get_settings()
        self._client = client or MilvusClient(uri=self._settings.milvus_uri)
        self.text_collection = self._settings.text_collection
        self.image_collection = self._settings.image_collection

    # --- collections -------------------------------------------------------------------------

    def ensure_collections(self) -> None:
        """Create both collections if they do not exist yet."""
        if not self._client.has_collection(self.text_collection):
            schema = self._base_schema(self._settings.text_embed_dim)
            schema.add_field("chunk_type", DataType.VARCHAR, max_length=16)
            schema.add_field("content", DataType.VARCHAR, max_length=MAX_CONTENT_BYTES)
            schema.add_field("chunk_index", DataType.INT64)
            self._create(self.text_collection, schema)

        if not self._client.has_collection(self.image_collection):
            schema = self._base_schema(self._settings.image_embed_dim)
            schema.add_field("image_path", DataType.VARCHAR, max_length=1024)
            schema.add_field("caption", DataType.VARCHAR, max_length=MAX_CAPTION_BYTES)
            self._create(self.image_collection, schema)

    def drop_collections(self) -> None:
        """Drop both collections. Needed after changing an embedding model or dimension."""
        for name in (self.text_collection, self.image_collection):
            if self._client.has_collection(name):
                self._client.drop_collection(name)

    @staticmethod
    def _base_schema(dim: int):
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
        schema.add_field("page", DataType.INT64, nullable=True)
        schema.add_field("metadata", DataType.JSON)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
        return schema

    def _create(self, name: str, schema) -> None:
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding", index_type="HNSW", metric_type=METRIC, params=INDEX_PARAMS
        )
        # Strong consistency: a document is searchable as soon as its upload returns.
        self._client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",
        )

    # --- writes ------------------------------------------------------------------------------

    def insert_text_chunks(
        self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]
    ) -> int:
        """Insert text and table chunks with their embeddings. Returns the number inserted."""
        rows = [
            {
                "id": chunk.id,
                "doc_id": chunk.doc_id,
                "chunk_type": chunk.chunk_type.value,
                "content": chunk.content,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
                "embedding": list(embedding),
            }
            for chunk, embedding in self._paired(chunks, embeddings)
        ]
        return self._insert(self.text_collection, rows)

    def insert_image_chunks(
        self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]
    ) -> int:
        """Insert image chunks with their embeddings. Returns the number inserted."""
        rows = [
            {
                "id": chunk.id,
                "doc_id": chunk.doc_id,
                "image_path": chunk.image_path or "",
                "caption": chunk.content,
                "page": chunk.page,
                "metadata": chunk.metadata,
                "embedding": list(embedding),
            }
            for chunk, embedding in self._paired(chunks, embeddings)
        ]
        return self._insert(self.image_collection, rows)

    @staticmethod
    def _paired(chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]):
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        return zip(chunks, embeddings, strict=True)

    def _insert(self, collection: str, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        return self._client.insert(collection_name=collection, data=rows)["insert_count"]

    def delete_document(self, doc_id: str) -> None:
        """Remove every chunk of a document from both collections."""
        for name in (self.text_collection, self.image_collection):
            self._client.delete(collection_name=name, filter=doc_filter([doc_id]))

    # --- reads -------------------------------------------------------------------------------

    def search_text(
        self, embedding: Sequence[float], top_k: int, doc_ids: Sequence[str] | None = None
    ) -> list[RetrievedChunk]:
        hits = self._search(self.text_collection, embedding, top_k, doc_ids, TEXT_FIELDS)
        return [
            RetrievedChunk(
                id=entity["id"],
                doc_id=entity["doc_id"],
                chunk_type=ChunkType(entity["chunk_type"]),
                content=entity["content"],
                page=entity.get("page"),
                chunk_index=entity["chunk_index"],
                metadata=entity.get("metadata") or {},
                similarity_score=score,
            )
            for entity, score in hits
        ]

    def search_images(
        self, embedding: Sequence[float], top_k: int, doc_ids: Sequence[str] | None = None
    ) -> list[RetrievedChunk]:
        hits = self._search(self.image_collection, embedding, top_k, doc_ids, IMAGE_FIELDS)
        return [
            RetrievedChunk(
                id=entity["id"],
                doc_id=entity["doc_id"],
                chunk_type=ChunkType.IMAGE,
                content=entity["caption"],
                page=entity.get("page"),
                image_path=entity["image_path"] or None,
                metadata=entity.get("metadata") or {},
                similarity_score=score,
            )
            for entity, score in hits
        ]

    def _search(
        self,
        collection: str,
        embedding: Sequence[float],
        top_k: int,
        doc_ids: Sequence[str] | None,
        fields: list[str],
    ) -> list[tuple[dict[str, Any], float]]:
        """Top-k hits as (entity, cosine similarity), best first."""
        if top_k <= 0:
            return []
        results = self._client.search(
            collection_name=collection,
            data=[list(embedding)],
            anns_field="embedding",
            limit=top_k,
            filter=doc_filter(doc_ids),
            output_fields=fields,
            search_params=SEARCH_PARAMS,
        )
        return [(hit["entity"], float(hit["distance"])) for hit in results[0]]

    def count_chunks(self, doc_id: str | None = None) -> dict[str, int]:
        """Row counts per collection, optionally for one document."""
        doc_ids = [doc_id] if doc_id else None
        counts = {}
        for name in (self.text_collection, self.image_collection):
            rows = self._client.query(
                collection_name=name, filter=doc_filter(doc_ids), output_fields=["count(*)"]
            )
            counts[name] = rows[0]["count(*)"]
        return counts


@lru_cache(maxsize=1)
def get_milvus_store() -> MilvusStore:
    """The shared store for the app, with collections created on first use."""
    store = MilvusStore()
    store.ensure_collections()
    return store
