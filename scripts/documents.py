"""Manage documents from the command line.

uv run python scripts/documents.py ingest path/to/file.pdf
uv run python scripts/documents.py ingest file.pdf --chunker llm --chunk-param window_chars=4000
uv run python scripts/documents.py list
uv run python scripts/documents.py delete <document-id>
"""

import argparse
import sys
from pathlib import Path

from ragdoc.console import use_utf8_output
from ragdoc.ingestion.parser import UnsupportedFileTypeError
from ragdoc.schemas import IngestionProgress, parse_chunking
from ragdoc.services.document_service import (
    DocumentInfo,
    DocumentService,
    DuplicateDocumentError,
    IngestionCancelled,
)


def describe(document: DocumentInfo) -> str:
    counts = (
        f"{document.n_text_chunks} text, {document.n_table_chunks} table, "
        f"{document.n_image_chunks} image"
    )
    ocr = "ocr on " if document.ocr_used else "ocr off"
    line = (
        f"{document.id}  {document.status.value:<10}  {counts:<28}  {ocr}  "
        f"{document.chunker:<9}  {document.filename}"
    )
    return f"{line}\n    error: {document.error}" if document.error else line


def main() -> int:
    use_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    ingest = commands.add_parser("ingest")
    ingest.add_argument("path", type=Path)
    ocr = ingest.add_mutually_exclusive_group()
    ocr.add_argument("--ocr", dest="ocr", action="store_true", default=None, help="OCR on")
    ocr.add_argument("--no-ocr", dest="ocr", action="store_false", help="OCR off")
    ingest.add_argument(
        "--chunker",
        choices=["recursive", "semantic", "llm"],
        help="chunking strategy (default: the CHUNKER setting)",
    )
    ingest.add_argument(
        "--chunk-param",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="override one parameter of the strategy, e.g. chunk_size=800; repeatable",
    )
    commands.add_parser("delete").add_argument("document_id")
    args = parser.parse_args()

    service = DocumentService()

    def show(progress: IngestionProgress) -> None:
        counts = (
            f" ({progress.current} of {progress.total} done)"
            if progress.total and progress.stage != "parse"
            else ""
        )
        print(f"- {progress.message}{counts}  [{progress.elapsed_s:.0f} s]", flush=True)

    if args.command == "list":
        documents = service.list_documents()
        print("\n".join(describe(d) for d in documents) if documents else "No documents.")
        return 0

    if args.command == "ingest":
        strategy = args.chunker or service.default_chunker
        params = service.chunking_defaults[strategy].params
        for item in args.chunk_param:
            name, _, value = item.partition("=")
            if name not in params:
                parser.error(f"{strategy} chunking has no parameter {name!r}: {', '.join(params)}")
            params[name] = value
        try:
            chunking = parse_chunking(strategy, params)
        except ValueError as exc:
            parser.error(str(exc))
        try:
            document = service.ingest(
                args.path.name,
                args.path.read_bytes(),
                on_progress=show,
                ocr=args.ocr,
                chunking=chunking,
            )
        except (UnsupportedFileTypeError, DuplicateDocumentError, IngestionCancelled) as exc:
            print(exc)
            return 1
        print(describe(document))
        print("    chunking: " + ", ".join(f"{k}={v}" for k, v in chunking.params.items()))
        if document.chunk_fallbacks:
            print(f"    {document.chunk_fallbacks} part(s) split by size: model reply unusable")
        if document.timings:
            print(
                "    "
                + ", ".join(f"{k[:-3]} {v / 1000:.1f} s" for k, v in document.timings.items())
            )
        return 0 if document.status.value == "ready" else 1

    deleted = service.delete(args.document_id)
    print("Deleted." if deleted else "No such document.")
    return 0 if deleted else 1


if __name__ == "__main__":
    sys.exit(main())
