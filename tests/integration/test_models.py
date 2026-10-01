"""Real model calls: Ollama on the Mac for text, CLIP in-process for images."""

import pytest
from langchain_core.messages import HumanMessage
from PIL import Image

from ragdoc.config import get_settings
from ragdoc.llm.chat import get_chat_model
from ragdoc.llm.embeddings import get_text_embedder
from ragdoc.llm.image_embeddings import get_image_embedder

pytestmark = pytest.mark.integration


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return dot / norm


def test_text_embedder_returns_configured_dimension():
    embedder = get_text_embedder()
    query = embedder.embed_query("How much revenue did the company make?")
    related, unrelated = embedder.embed_documents(
        ["Total revenue for the year was 4.2 million dollars.", "Cats sleep most of the day."]
    )
    assert len(query) == len(related) == get_settings().text_embed_dim
    assert cosine(query, related) > cosine(query, unrelated)


def test_chat_model_answers():
    answer = get_chat_model().invoke([HumanMessage("Reply with the single word: pong")])
    assert "pong" in answer.text.lower()


def test_image_embedder_puts_text_and_images_in_one_space(tmp_path):
    red, blue = tmp_path / "red.png", tmp_path / "blue.png"
    Image.new("RGB", (224, 224), (255, 0, 0)).save(red)
    Image.new("RGB", (224, 224), (0, 0, 255)).save(blue)

    embedder = get_image_embedder()
    red_vector, blue_vector = embedder.embed_images([red, blue])
    query = embedder.embed_query("a solid red image")

    assert len(red_vector) == len(query) == get_settings().image_embed_dim
    assert cosine(query, red_vector) > cosine(query, blue_vector)
    assert embedder.embed_images([]) == []
