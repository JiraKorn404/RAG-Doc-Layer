"""Check that the remote Ollama server is ready for this project.

Verifies, using the values in `.env`:
  1. the server is reachable
  2. the chat, vision and text embedding models are pulled
  3. the embedding model returns vectors of `TEXT_EMBED_DIM`
  4. the vision model can actually read an image

Run: uv run python scripts/check_ollama.py
"""

import base64
import struct
import sys
import time
import zlib

import httpx
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

from ragdoc.config import get_settings


def solid_png(rgb: tuple[int, int, int], size: int = 64) -> bytes:
    """A solid-colour PNG, built by hand so the check needs no imaging library."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    row = b"\x00" + bytes(rgb) * size
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(row * size))
        + chunk(b"IEND", b"")
    )


def check_reachable(base_url: str, timeout: float) -> set[str]:
    response = httpx.get(f"{base_url}/api/tags", timeout=timeout)
    response.raise_for_status()
    return {model["name"] for model in response.json()["models"]}


def main() -> int:
    settings = get_settings()
    failures: list[str] = []

    def report(ok: bool, label: str, detail: str = "") -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}{f' - {detail}' if detail else ''}")
        if not ok:
            failures.append(label)

    print(f"Ollama: {settings.ollama_base_url}\n")

    try:
        available = check_reachable(settings.ollama_base_url, timeout=10)
    except httpx.HTTPError as exc:
        report(False, "server reachable", f"{type(exc).__name__}: {exc}")
        print(
            "\nCheck that Tailscale is connected on both machines and that Ollama on the Mac "
            "was started with OLLAMA_HOST=0.0.0.0."
        )
        return 1
    report(True, "server reachable", f"{len(available)} models available")

    required = {
        "CHAT_MODEL": settings.chat_model,
        "VISION_MODEL": settings.vision_model,
        "TEXT_EMBED_MODEL": settings.text_embed_model,
    }
    for name, model in required.items():
        report(model in available, f"{name} pulled", model)
    if failures:
        print("\nPull the missing models on the Mac with `ollama pull <model>`.")
        return 1

    try:
        embeddings = OllamaEmbeddings(
            model=settings.text_embed_model,
            base_url=settings.ollama_base_url,
            keep_alive=settings.ollama_keep_alive_s,
            client_kwargs={"timeout": settings.ollama_timeout_s},
        )
        started = time.perf_counter()
        vector = embeddings.embed_query("What does the quarterly report say about revenue?")
        elapsed = time.perf_counter() - started
        report(
            len(vector) == settings.text_embed_dim,
            "embedding dimension",
            f"got {len(vector)}, expected {settings.text_embed_dim} ({elapsed:.1f}s)",
        )
    except Exception as exc:  # noqa: BLE001 - report any client failure and keep checking
        report(False, "embedding dimension", f"{type(exc).__name__}: {exc}")

    try:
        vision = ChatOllama(
            model=settings.vision_model,
            base_url=settings.ollama_base_url,
            keep_alive=settings.ollama_keep_alive_s,
            temperature=0,
            client_kwargs={"timeout": settings.ollama_timeout_s},
        )
        image_b64 = base64.b64encode(solid_png((255, 0, 0))).decode()
        message = HumanMessage(
            content=[
                {"type": "text", "text": "What colour fills this image? Answer with one word."},
                {"type": "image_url", "image_url": f"data:image/png;base64,{image_b64}"},
            ]
        )
        started = time.perf_counter()
        answer = vision.invoke([message]).text.strip()
        elapsed = time.perf_counter() - started
        report("red" in answer.lower(), "vision model reads images", f"{answer!r} ({elapsed:.1f}s)")
    except Exception as exc:  # noqa: BLE001
        report(False, "vision model reads images", f"{type(exc).__name__}: {exc}")

    print(f"\n{'All checks passed.' if not failures else f'{len(failures)} check(s) failed.'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
