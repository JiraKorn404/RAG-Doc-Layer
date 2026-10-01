"""Run a retrieval query and print the top text and image chunks with their scores.

uv run python scripts/search.py "How much revenue did the company make in 2024?"
uv run python scripts/search.py "revenue chart" --text-k 3 --image-k 2
"""

import argparse
import sys

from ragdoc.console import use_utf8_output
from ragdoc.retrieval.retriever import get_retriever
from ragdoc.schemas import RetrievedChunk

PREVIEW_CHARS = 300


def describe(rank: int, chunk: RetrievedChunk) -> str:
    source = chunk.metadata.get("filename", chunk.doc_id)
    page = f"p.{chunk.page}" if chunk.page is not None else "no page"
    header = (
        f"  {rank}. similarity {chunk.similarity_score:.4f}  "
        f"[{chunk.chunk_type.value}]  {source}, {page}"
    )
    if chunk.image_path:
        body = f"{chunk.image_path}  caption: {chunk.content or '(none)'}"
    else:
        body = " ".join(chunk.content.split())
        if len(body) > PREVIEW_CHARS:
            body = body[:PREVIEW_CHARS] + "..."
    return f"{header}\n     {body}"


def main() -> int:
    use_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query")
    parser.add_argument("--text-k", type=int, default=None, help="default: TEXT_TOP_K")
    parser.add_argument("--image-k", type=int, default=None, help="default: IMAGE_TOP_K")
    args = parser.parse_args()

    result = get_retriever().retrieve(args.query, text_top_k=args.text_k, image_top_k=args.image_k)

    print(f"Query: {result.query}\n")
    print(f"Text and table chunks ({len(result.text)}):")
    print("\n".join(describe(i, c) for i, c in enumerate(result.text, 1)) or "  (none)")
    print(f"\nImage chunks ({len(result.images)}):")
    print("\n".join(describe(i, c) for i, c in enumerate(result.images, 1)) or "  (none)")
    timings = result.timings
    if timings:
        print(
            f"\nEmbedding {timings['embed_ms']:.0f} ms, search {timings['search_ms']:.0f} ms, "
            f"total {timings['total_ms']:.0f} ms"
        )
    print("\nText and image scores come from different models and are not comparable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
