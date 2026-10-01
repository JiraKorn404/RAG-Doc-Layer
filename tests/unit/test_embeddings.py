import pytest
from langchain_core.embeddings import Embeddings

from ragdoc.llm.embeddings import EmbeddingDimensionError, TextEmbedder


class FakeEmbeddings(Embeddings):
    def __init__(self, dim: int):
        self.dim = dim
        self.seen: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.seen.extend(texts)
        return [[0.0] * self.dim for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.seen.append(text)
        return [0.0] * self.dim


def make_embedder(inner: FakeEmbeddings, dim: int = 4, instruction: str = "find passages"):
    return TextEmbedder(inner, dim=dim, model="fake", query_instruction=instruction)


def test_query_gets_instruction_prefix():
    inner = FakeEmbeddings(4)
    make_embedder(inner).embed_query("what is x?")
    assert inner.seen == ["Instruct: find passages\nQuery: what is x?"]


def test_documents_are_not_prefixed():
    inner = FakeEmbeddings(4)
    make_embedder(inner).embed_documents(["a", "b"])
    assert inner.seen == ["a", "b"]


def test_no_instruction_leaves_query_unchanged():
    inner = FakeEmbeddings(4)
    make_embedder(inner, instruction="").embed_query("q")
    assert inner.seen == ["q"]


def test_empty_document_list_skips_the_model():
    inner = FakeEmbeddings(4)
    assert make_embedder(inner).embed_documents([]) == []
    assert inner.seen == []


def test_dimension_mismatch_raises():
    embedder = make_embedder(FakeEmbeddings(3), dim=4)
    with pytest.raises(EmbeddingDimensionError):
        embedder.embed_query("q")
    with pytest.raises(EmbeddingDimensionError):
        embedder.embed_documents(["a"])
