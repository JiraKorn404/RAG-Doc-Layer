"""Document parsing: a file becomes an ordered list of text, table and image elements."""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from ragdoc.config import Settings, get_settings
from ragdoc.schemas import ChunkType

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
PLAIN_TEXT_EXTENSIONS = {".txt"}
SUPPORTED_EXTENSIONS = (
    {".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".md"} | IMAGE_EXTENSIONS | PLAIN_TEXT_EXTENSIONS
)

MAX_CAPTION_CHARS = 2000

# Called before each batch of pages is parsed, with (first page, last page, total pages),
# 1-based. Raising from it stops the parse: that is how an ingestion is cancelled.
PagesCallback = Callable[[int, int, int], None]


class UnsupportedFileTypeError(ValueError):
    pass


class ParseTimeoutError(TimeoutError):
    pass


class ParsedElement(BaseModel):
    """One piece of a parsed document.

    `content` is the text for `text` elements and the Markdown table for `table` elements;
    it is empty for `image` elements, whose file is at `image_path`.
    """

    type: ChunkType
    content: str = ""
    caption: str = ""
    page: int | None = None
    image_path: Path | None = None


class DocumentParser(Protocol):
    def parse(
        self,
        path: Path,
        image_dir: Path,
        *,
        ocr: bool = False,
        on_pages: PagesCallback | None = None,
    ) -> list[ParsedElement]:
        """Parse `path`, saving any extracted pictures under `image_dir`.

        With `ocr`, text inside pictures and scanned pages is read too (slower). `on_pages` is
        called before each batch of pages, for formats that have pages.
        """
        ...


def pdf_page_count(path: Path) -> int | None:
    """Number of pages, or None if the file cannot be opened as a PDF."""
    import pypdfium2

    try:
        document = pypdfium2.PdfDocument(str(path))
    except Exception:  # noqa: BLE001 - let Docling report what is wrong with the file
        return None
    try:
        return len(document)
    finally:
        document.close()


class DoclingParser:
    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        # One converter per OCR mode, each built on first use.
        self._converters: dict[bool, object] = {}

    def parse(
        self,
        path: Path,
        image_dir: Path,
        *,
        ocr: bool = False,
        on_pages: PagesCallback | None = None,
    ) -> list[ParsedElement]:
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"{suffix or 'no extension'} is not supported. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        # Docling has no plain-text input format.
        if suffix in PLAIN_TEXT_EXTENSIONS:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            return [ParsedElement(type=ChunkType.TEXT, content=text)] if text else []

        converter = self._get_converter(ocr)

        if suffix == ".pdf" and (n_pages := pdf_page_count(path)):
            return self._parse_pdf_in_batches(converter, path, image_dir, n_pages, on_pages)

        document = converter.convert(path).document

        # An uploaded picture is itself the image; Docling only contributes any text it reads
        # from it (OCR), not sub-crops of the same picture.
        if suffix in IMAGE_EXTENSIONS:
            elements = self._elements(document, image_dir, with_pictures=False)
            return [*elements, self._whole_image(path, image_dir)]
        return self._elements(document, image_dir, with_pictures=True)

    def _parse_pdf_in_batches(
        self,
        converter,
        path: Path,
        image_dir: Path,
        n_pages: int,
        on_pages: PagesCallback | None,
    ) -> list[ParsedElement]:
        """Parse a few pages at a time. A Docling conversion cannot be interrupted, so the
        batches are what make progress reporting, cancellation and the time limit possible."""
        batch = self._settings.parse_page_batch
        deadline = time.monotonic() + self._settings.parse_timeout_s
        elements: list[ParsedElement] = []
        n_images = 0
        for first in range(1, n_pages + 1, batch):
            last = min(first + batch - 1, n_pages)
            if time.monotonic() > deadline:
                raise ParseTimeoutError(
                    f"Parsing took longer than {self._settings.parse_timeout_s:.0f} s "
                    f"(stopped before page {first} of {n_pages})."
                )
            if on_pages is not None:
                on_pages(first, last, n_pages)
            document = converter.convert(path, page_range=(first, last)).document
            new = self._elements(document, image_dir, with_pictures=True, first_image=n_images + 1)
            n_images += sum(element.type is ChunkType.IMAGE for element in new)
            elements.extend(new)
        return elements

    def _get_converter(self, ocr: bool):
        if ocr not in self._converters:
            # Imported here so that importing ragdoc does not load Docling's models.
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import (
                DocumentConverter,
                ImageFormatOption,
                PdfFormatOption,
            )

            options = PdfPipelineOptions()
            options.do_ocr = ocr
            options.generate_picture_images = True
            options.images_scale = self._settings.pdf_image_scale
            # A backstop for one batch that hangs; the overall limit is enforced between batches.
            options.document_timeout = self._settings.parse_timeout_s
            self._converters[ocr] = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=options),
                    InputFormat.IMAGE: ImageFormatOption(pipeline_options=options),
                }
            )
        return self._converters[ocr]

    def _elements(
        self, document, image_dir: Path, *, with_pictures: bool, first_image: int = 1
    ) -> list[ParsedElement]:
        from docling_core.types.doc import (
            DocItemLabel,
            ListItem,
            PictureItem,
            SectionHeaderItem,
            TableItem,
            TextItem,
        )

        elements: list[ParsedElement] = []
        # Consecutive text items on one page are merged, so the chunker sees running text.
        buffer: list[str] = []
        buffer_page: int | None = None
        image_number = first_image

        def flush() -> None:
            if buffer:
                elements.append(
                    ParsedElement(
                        type=ChunkType.TEXT, content="\n\n".join(buffer), page=buffer_page
                    )
                )
                buffer.clear()

        for item, _level in document.iterate_items():
            page = item.prov[0].page_no if getattr(item, "prov", None) else None

            if isinstance(item, TableItem):
                flush()
                markdown = item.export_to_markdown(doc=document).strip()
                if markdown:
                    elements.append(
                        ParsedElement(
                            type=ChunkType.TABLE,
                            content=markdown,
                            caption=item.caption_text(document)[:MAX_CAPTION_CHARS],
                            page=page,
                        )
                    )

            elif isinstance(item, PictureItem):
                if not with_pictures:
                    continue
                image = item.get_image(document)
                if image is None or min(image.size) < self._settings.min_image_side_px:
                    continue
                image_dir.mkdir(parents=True, exist_ok=True)
                image_path = image_dir / f"image_{image_number:03d}.png"
                image_number += 1
                image.save(image_path, format="PNG")
                elements.append(
                    ParsedElement(
                        type=ChunkType.IMAGE,
                        caption=item.caption_text(document)[:MAX_CAPTION_CHARS],
                        page=page,
                        image_path=image_path,
                    )
                )

            elif isinstance(item, TextItem):
                text = item.text.strip()
                if not text:
                    continue
                if page != buffer_page:
                    flush()
                    buffer_page = page
                if item.label == DocItemLabel.TITLE:
                    text = f"# {text}"
                elif isinstance(item, SectionHeaderItem):
                    text = f"## {text}"
                elif isinstance(item, ListItem):
                    text = f"- {text}"
                buffer.append(text)

        flush()
        return elements

    @staticmethod
    def _whole_image(path: Path, image_dir: Path) -> ParsedElement:
        from PIL import Image

        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / "image_001.png"
        with Image.open(path) as image:
            image.convert("RGB").save(image_path, format="PNG")
        return ParsedElement(type=ChunkType.IMAGE, page=1, image_path=image_path)


def get_parser(settings: Settings | None = None) -> DocumentParser:
    return DoclingParser(settings)
