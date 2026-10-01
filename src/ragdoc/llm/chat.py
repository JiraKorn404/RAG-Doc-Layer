"""Chat and vision model factories."""

from typing import Any

from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama

from ragdoc.config import Settings, get_settings


def _build(model: str, settings: Settings, temperature: float, **kwargs: Any) -> ChatOllama:
    return ChatOllama(
        model=model,
        base_url=settings.ollama_base_url,
        keep_alive=settings.ollama_keep_alive_s,
        temperature=temperature,
        client_kwargs={"timeout": settings.ollama_timeout_s},
        **kwargs,
    )


def get_chat_model(
    settings: Settings | None = None, *, temperature: float = 0.0, **kwargs: Any
) -> ChatOllama:
    """The main chat model. Extra kwargs go to `ChatOllama` (e.g. `reasoning=True`)."""
    settings = settings or get_settings()
    return _build(settings.chat_model, settings, temperature, **kwargs)


def get_chunking_model(settings: Settings | None = None) -> ChatOllama:
    """The chat model as used to pick chunk boundaries: no thinking, a short output limit."""
    settings = settings or get_settings()
    return _build(
        settings.chat_model,
        settings,
        0.0,
        reasoning=False,
        num_predict=settings.chunking_max_tokens,
    )


def text_of(message: BaseMessage) -> str:
    """The text of a model reply, whether its content is a string or a list of blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block) for block in content
    )


def get_vision_model(
    settings: Settings | None = None, *, temperature: float = 0.0, **kwargs: Any
) -> ChatOllama:
    """The model used when images are part of the prompt."""
    settings = settings or get_settings()
    return _build(settings.vision_model, settings, temperature, **kwargs)
