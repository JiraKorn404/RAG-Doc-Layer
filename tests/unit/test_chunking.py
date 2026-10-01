from pathlib import Path

import pytest

from ragdoc.config import Settings
from ragdoc.ingestion.chunking.factory import get_chunker
from ragdoc.ingestion.chunking.recursive import RecursiveChunker, split_table
from ragdoc.ingestion.parser import ParsedElement
from ragdoc.schemas import ChunkType


def make_chunker(chunk_size=100, chunk_overlap=20, table_max_chars=200) -> RecursiveChunker:
    return RecursiveChunker(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, table_max_chars=table_max_chars
    )


def make_table(n_rows: int) -> str:
    rows = [f"| row {i:03d} | value {i:03d} |" for i in range(n_rows)]
    return "\n".join(["| name | value |", "| --- | --- |", *rows])


def test_long_text_is_split_within_chunk_size():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = make_chunker().chunk(
        [ParsedElement(type=ChunkType.TEXT, content=text, page=4)], doc_id="d"
    )
    assert len(chunks) > 1
    assert all(len(chunk.content) <= 100 for chunk in chunks)
    assert all(chunk.chunk_type is ChunkType.TEXT and chunk.page == 4 for chunk in chunks)


def test_chunk_index_runs_across_elements_and_metadata_is_copied():
    elements = [
        ParsedElement(type=ChunkType.TEXT, content="first page text", page=1),
        ParsedElement(type=ChunkType.TABLE, content=make_table(2), page=1),
        ParsedElement(type=ChunkType.TEXT, content="second page text", page=2),
    ]
    chunks = make_chunker().chunk(elements, doc_id="doc-1", metadata={"filename": "f.pdf"})
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert [chunk.page for chunk in chunks] == [1, 1, 2]
    assert all(
        chunk.doc_id == "doc-1" and chunk.metadata == {"filename": "f.pdf"} for chunk in chunks
    )
    chunks[0].metadata["x"] = 1
    assert "x" not in chunks[1].metadata


def test_small_table_stays_whole_even_if_larger_than_chunk_size():
    table = make_table(5)
    assert len(table) > 100
    [chunk] = make_chunker(table_max_chars=1000).chunk(
        [ParsedElement(type=ChunkType.TABLE, content=table)], doc_id="d"
    )
    assert chunk.chunk_type is ChunkType.TABLE
    assert chunk.content == table


def test_large_table_is_split_by_rows_with_header_repeated():
    table = make_table(30)
    parts = split_table(table, max_chars=200)
    assert len(parts) > 1
    assert all(part.startswith("| name | value |\n| --- | --- |\n") for part in parts)
    assert all(len(part) <= 200 for part in parts)
    rows = [line for part in parts for line in part.splitlines()[2:]]
    assert rows == table.splitlines()[2:]


def test_table_caption_is_put_on_every_part():
    element = ParsedElement(type=ChunkType.TABLE, content=make_table(30), caption="Table 1: values")
    chunks = make_chunker(table_max_chars=200).chunk([element], doc_id="d")
    assert len(chunks) > 1
    assert all(chunk.content.startswith("Table 1: values\n\n| name |") for chunk in chunks)
    assert all(len(chunk.content) <= 200 for chunk in chunks)


def test_row_longer_than_the_limit_is_still_split():
    table = "\n".join(["| a |", "| --- |", "| " + "x " * 300 + "|"])
    chunks = make_chunker(table_max_chars=200).chunk(
        [ParsedElement(type=ChunkType.TABLE, content=table)], doc_id="d"
    )
    assert len(chunks) > 1
    assert all(len(chunk.content) <= 200 for chunk in chunks)


def test_image_becomes_one_chunk_with_caption_as_content():
    element = ParsedElement(
        type=ChunkType.IMAGE, caption="Figure 1", page=2, image_path=Path("images/a.png")
    )
    [chunk] = make_chunker().chunk([element], doc_id="d")
    assert chunk.chunk_type is ChunkType.IMAGE
    assert chunk.content == "Figure 1"
    assert Path(chunk.image_path) == Path("images/a.png")


def test_factory_builds_configured_chunker():
    assert isinstance(get_chunker(Settings(_env_file=None)), RecursiveChunker)


def test_factory_rejects_unknown_chunker():
    with pytest.raises(ValueError, match="Unknown chunker"):
        get_chunker(Settings(_env_file=None, chunker="nope"))
