"""Turning state into model input, and model output back into state."""

import base64
import io
import re
from collections.abc import Collection
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ragdoc.agent.state import HistoryMessage
from ragdoc.config import Settings
from ragdoc.schemas import RetrievalResult, RetrievedChunk

# [1], [2][3] and [1, 2]
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def history_messages(history: list[HistoryMessage]) -> list[BaseMessage]:
    return [
        HumanMessage(item["content"]) if item["role"] == "user" else AIMessage(item["content"])
        for item in history
    ]


def source_label(number: int, chunk: RetrievedChunk) -> str:
    """The heading a source is shown under, e.g. `[2] report.pdf, page 3, table`."""
    parts = [chunk.metadata.get("filename", "document")]
    if chunk.page is not None:
        parts.append(f"page {chunk.page}")
    parts.append(chunk.chunk_type.value)
    return f"[{number}] {', '.join(parts)}"


def image_data_url(settings: Settings, relative_path: str) -> str | None:
    """The image as a PNG data URL, downscaled; None if the file is gone."""
    from PIL import Image

    path = settings.resolve_data_path(relative_path)
    if not path.is_file():
        return None
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((settings.max_image_side_px, settings.max_image_side_px))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def source_blocks(retrieval: RetrievalResult, settings: Settings) -> list[dict[str, Any]]:
    """Message content blocks presenting every retrieved chunk as a numbered source.

    Numbers follow `retrieval.chunks`: text and tables first, then images.
    """
    blocks: list[dict[str, Any]] = []
    for number, chunk in enumerate(retrieval.chunks, start=1):
        label = source_label(number, chunk)
        if chunk.image_path is None:
            blocks.append({"type": "text", "text": f"{label}\n{chunk.content}"})
            continue
        caption = f" Its caption: {chunk.content}" if chunk.content else ""
        url = image_data_url(settings, chunk.image_path)
        if url is None:
            text = f"{label}\nThe image file for source [{number}] is no longer available.{caption}"
            blocks.append({"type": "text", "text": text})
        else:
            # Said explicitly: with text sources alongside, the model otherwise tends to treat
            # the picture as absent and answer from the text alone.
            text = (
                f"{label}\nSource [{number}] is the image attached directly after this line."
                f"{caption}"
            )
            blocks.append({"type": "text", "text": text})
            blocks.append({"type": "image_url", "image_url": url})
    return blocks


def cited_chunk_ids(answer: str, retrieval: RetrievalResult) -> list[str]:
    """Ids of the chunks the answer cites, in order of first citation."""
    chunks = retrieval.chunks
    ids: list[str] = []
    for group in CITATION.findall(answer):
        for number in (int(part) for part in group.split(",")):
            if 1 <= number <= len(chunks) and chunks[number - 1].id not in ids:
                ids.append(chunks[number - 1].id)
    return ids


def usage_of(message: BaseMessage | None) -> dict[str, int]:
    """Token counts of one model call, in the shape `AgentState.usage` accumulates."""
    usage = getattr(message, "usage_metadata", None) or {}
    return {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
    }


def text_of(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block) for block in content
    )


def reasoning_of(message: BaseMessage) -> str:
    return message.additional_kwargs.get("reasoning_content") or ""


def invoke_decision(
    model: BaseChatModel, messages: list[BaseMessage], *, label: str, choices: Collection[str]
) -> tuple[str | None, str, dict[str, int]]:
    """Ask for a short reason and a one-word decision. Returns (choice, reason, usage); the
    choice is None if the reply did not contain one of `choices`, so the caller can fall back
    to a safe default.

    The prompt must ask for two lines, `Reason: ...` then `<label>: <choice>`. This is plain
    text on purpose: with JSON-constrained output, `gemma4:e4b-mlx` sometimes stops after the
    first field and emits whitespace until the output limit, losing the decision.
    """
    reply = model.invoke(messages)
    choice, reason = parse_decision(text_of(reply), label, choices)
    return choice, reason, usage_of(reply)


def parse_decision(text: str, label: str, choices: Collection[str]) -> tuple[str | None, str]:
    """Extract (choice, reason) from a `Reason: ... / <label>: <choice>` reply, tolerating
    Markdown emphasis, quotes and extra text. The last `<label>:` line wins."""
    decision = re.findall(rf"(?im)^[\s*_>#-]*{re.escape(label)}[\s*_]*:[\s*_\"'`]*([a-z_]+)", text)
    choice = decision[-1].lower() if decision else None
    if choice not in choices:
        choice = None
    reason = re.search(
        rf"(?ims)^[\s*_>#-]*reason[\s*_]*:[\s*_]*(.*?)\s*(?=^[\s*_>#-]*{re.escape(label)}[\s*_]*:|\Z)",
        text,
    )
    return choice, (reason.group(1).strip() if reason else "")
