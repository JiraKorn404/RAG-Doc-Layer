"""Semantic and LLM-based chunking, the per-upload configuration, and the factory."""

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from ragdoc.config import Settings
from ragdoc.ingestion.chunking.factory import get_chunker
from ragdoc.ingestion.chunking.llm import LLMChunker, parse_splits
from ragdoc.ingestion.chunking.recursive import RecursiveChunker
from ragdoc.ingestion.chunking.semantic import SemanticChunker
from ragdoc.ingestion.chunking.sentences import split_sentences
from ragdoc.ingestion.parser import ParsedElement
from ragdoc.schemas import (
    ChunkType,
    LLMChunking,
    RecursiveChunking,
    SemanticChunking,
    parse_chunking,
)

CATS = "Cats purr loudly at night. Cats sleep for most of the day. "
STOCKS = "Stocks fell sharply on Monday. Stocks recovered again by Friday."


def text_element(content: str, page: int = 1) -> ParsedElement:
    return ParsedElement(type=ChunkType.TEXT, content=content, page=page)


def contents(chunks) -> list[str]:
    return [chunk.content for chunk in chunks]


# --- sentences -----------------------------------------------------------------------------


def test_sentences_join_back_to_the_text():
    text = "## Results\n\nRevenue rose to 3.5 million. Costs fell (slightly).  Why? Nobody knows."
    sentences = split_sentences(text)
    assert "".join(sentences) == text
    assert [s.strip() for s in sentences] == [
        "## Results\n\nRevenue rose to 3.5 million.",
        "Costs fell (slightly).",
        "Why? Nobody knows.",
    ]


def test_short_fragments_stay_with_their_neighbours():
    assert split_sentences("1. Introduction to the annual report.") == [
        "1. Introduction to the annual report."
    ]
    assert split_sentences("   ") == []


# --- configuration -------------------------------------------------------------------------


def test_parameters_are_validated():
    with pytest.raises(ValidationError, match="overlap must be smaller"):
        RecursiveChunking(chunk_size=500, chunk_overlap=500)
    with pytest.raises(ValidationError, match="minimum chunk size must be smaller"):
        SemanticChunking(min_chunk_chars=600, max_chunk_chars=500)
    with pytest.raises(ValidationError, match="must not exceed the maximum"):
        LLMChunking(target_chunk_chars=3000, max_chunk_chars=2000)
    with pytest.raises(ValidationError):
        SemanticChunking(breakpoint_percentile=100)


def test_parse_chunking_builds_the_strategy_model_from_strings():
    config = parse_chunking("semantic", {"breakpoint_percentile": "85"})
    assert isinstance(config, SemanticChunking)
    assert config.params == {
        "breakpoint_percentile": 85,
        "buffer_sentences": 1,
        "min_chunk_chars": 200,
        "max_chunk_chars": 2000,
    }
    with pytest.raises(ValueError, match="Unknown chunker"):
        parse_chunking("nope")


def test_settings_supply_each_strategys_defaults():
    settings = Settings(_env_file=None, chunker="llm", llm_chunk_window_chars=4000, chunk_size=800)
    defaults = settings.chunking_defaults()
    assert list(defaults) == ["recursive", "semantic", "llm"]
    assert defaults["recursive"].chunk_size == 800
    assert settings.default_chunking() == LLMChunking(window_chars=4000)


def test_out_of_range_default_is_rejected_when_settings_load():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, semantic_breakpoint_percentile=20)


# --- semantic ------------------------------------------------------------------------------


class TopicEmbedder:
    """Embeds a sentence by its topic word, so the distance between topics is known."""

    def __init__(self):
        self.batches: list[int] = []

    def embed_documents(self, texts):
        self.batches.append(len(texts))
        return [[1.0, 0.0] if "Cats" in text else [0.0, 1.0] for text in texts]


def semantic(embedder=None, **overrides) -> SemanticChunker:
    options = {
        "breakpoint_percentile": 50,
        "buffer_sentences": 0,
        "min_chunk_chars": 0,
        "max_chunk_chars": 2000,
        "table_max_chars": 500,
        "batch_size": 3,
    }
    return SemanticChunker(embedder=embedder or TopicEmbedder(), **{**options, **overrides})


def test_semantic_splits_where_the_topic_changes():
    embedder = TopicEmbedder()
    progress = []
    chunks = semantic(embedder).chunk(
        [text_element(CATS + STOCKS, page=3)],
        doc_id="d",
        checkpoint=lambda *args: progress.append(args),
    )
    assert contents(chunks) == [CATS.strip(), STOCKS]
    assert all(chunk.page == 3 and chunk.chunk_type is ChunkType.TEXT for chunk in chunks)
    # Four sentences embedded in batches of three, with a checkpoint before each batch.
    assert embedder.batches == [3, 1]
    assert progress == [
        ("Chunking: embedding sentences", 0, 4),
        ("Chunking: embedding sentences", 3, 4),
    ]


def test_semantic_merges_chunks_below_the_minimum():
    stocks = f"{STOCKS} {STOCKS}"
    text = f"Cats purr loudly at night. {stocks} Cats sleep for most of the day."
    assert len(semantic().chunk([text_element(text)], doc_id="d")) == 3
    # The lone first and last sentences are too short to stand alone.
    [chunk] = semantic(min_chunk_chars=40).chunk([text_element(text)], doc_id="d")
    assert chunk.content == text


def test_semantic_cuts_chunks_over_the_maximum():
    text = CATS * 20
    chunks = semantic(max_chunk_chars=300).chunk([text_element(text)], doc_id="d")
    sizes = [len(chunk.content) for chunk in chunks]
    assert len(chunks) > 1 and max(sizes) <= 300
    # Cut between sentences, into parts of similar size, with nothing lost.
    assert all(chunk.content.endswith(".") for chunk in chunks)
    assert min(sizes) > 150
    assert " ".join(contents(chunks)) == text.strip()


def test_a_single_sentence_over_the_maximum_is_still_cut():
    text = "Cats " + "purr and " * 100 + "sleep. " + STOCKS
    chunks = semantic(max_chunk_chars=300).chunk([text_element(text)], doc_id="d")
    assert len(chunks) > 3 and all(len(chunk.content) <= 300 for chunk in chunks)


def test_semantic_keeps_pages_apart_and_does_not_embed_single_sentences():
    embedder = TopicEmbedder()
    elements = [
        text_element("Cats purr loudly at night.", page=1),
        ParsedElement(type=ChunkType.TABLE, content="| a |\n| --- |\n| 1 |", page=1),
        text_element(STOCKS, page=2),
    ]
    chunks = semantic(embedder).chunk(elements, doc_id="d")
    assert [(chunk.chunk_type.value, chunk.page) for chunk in chunks] == [
        ("text", 1),
        ("table", 1),
        ("text", 2),
    ]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert sum(embedder.batches) == 2  # only the two sentences of page 2


def test_cancelling_from_the_checkpoint_stops_semantic_chunking():
    class Cancelled(Exception):
        pass

    def stop_at_second_batch(_message, current, _total):
        if current:
            raise Cancelled

    embedder = TopicEmbedder()
    with pytest.raises(Cancelled):
        semantic(embedder).chunk(
            [text_element(CATS + STOCKS)], doc_id="d", checkpoint=stop_at_second_batch
        )
    assert embedder.batches == [3]


# --- LLM-based -----------------------------------------------------------------------------


class ScriptedModel:
    """Returns the given replies in order and records what it was shown."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def invoke(self, messages):
        self.prompts.append(messages[-1].content)
        return AIMessage(self.replies.pop(0))


def llm(model, **overrides) -> LLMChunker:
    options = {
        "target_chunk_chars": 50,
        "max_chunk_chars": 2000,
        "window_chars": 6000,
        "table_max_chars": 500,
    }
    return LLMChunker(model=model, **{**options, **overrides})


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("Reason: topic changes.\nSplits: 3, 5", [2, 4]),
        ("**Reason:** x\n**Splits:** 3 and 5.", [2, 4]),
        ("Reason: one topic.\nSplits: none", []),
        ("Splits: 1, 3, 3, 2, 9, 5", [2, 4]),  # 1, repeats, decreasing and out of range dropped
        ("Splits: 2\nSplits: 4", [3]),  # the last line wins
        ("Reason: I think it should be split somewhere.", None),
        ("Reason: x\nSplits:", None),
        ("", None),
    ],
)
def test_parse_splits(reply, expected):
    assert parse_splits(reply, n_sentences=6) == expected


def test_llm_cuts_the_original_text_at_the_sentences_the_model_names():
    model = ScriptedModel("Reason: cats, then stocks.\nSplits: 3")
    progress = []
    chunker = llm(model)
    chunks = chunker.chunk(
        [text_element(CATS + STOCKS, page=2)],
        doc_id="d",
        checkpoint=lambda *args: progress.append(args),
    )
    assert contents(chunks) == [CATS.strip(), STOCKS]
    assert chunks[0].page == 2
    assert model.prompts == [
        "[1] Cats purr loudly at night.\n[2] Cats sleep for most of the day.\n"
        "[3] Stocks fell sharply on Monday.\n[4] Stocks recovered again by Friday."
    ]
    assert progress == [("Chunking: sentences read by the chat model", 0, 4)]
    assert chunker.n_fallbacks == 0


def test_llm_does_not_call_the_model_for_short_text():
    model = ScriptedModel()
    [chunk] = llm(model, target_chunk_chars=500).chunk([text_element(CATS + STOCKS)], doc_id="d")
    assert chunk.content == CATS + STOCKS
    assert model.prompts == []


def test_unreadable_reply_falls_back_to_plain_splitting_and_is_counted():
    chunker = llm(ScriptedModel("I would split this after the cats."))
    chunks = chunker.chunk([text_element(CATS + STOCKS)], doc_id="d")
    assert chunker.n_fallbacks == 1
    assert len(chunks) > 1 and all(len(chunk.content) <= 50 for chunk in chunks)
    assert " ".join(contents(chunks)) == CATS + STOCKS


def test_sentences_after_the_last_boundary_open_the_next_window():
    # Two sentences fit a window. The first reply closes "Cats purr"; "Cats sleep" is carried
    # over and shown again with the next sentence.
    model = ScriptedModel("Splits: 2", "Splits: 2", "Splits: none")
    chunks = llm(model, window_chars=65).chunk([text_element(CATS + STOCKS)], doc_id="d")
    assert [prompt.count("\n") + 1 for prompt in model.prompts] == [2, 2, 2]
    assert model.prompts[1].startswith("[1] Cats sleep for most of the day.\n[2] Stocks fell")
    assert contents(chunks) == [
        "Cats purr loudly at night.",
        "Cats sleep for most of the day.",
        STOCKS,
    ]


def test_llm_cuts_chunks_over_the_maximum():
    text = CATS * 20
    chunks = llm(ScriptedModel("Splits: none"), max_chunk_chars=300).chunk(
        [text_element(text)], doc_id="d"
    )
    assert len(chunks) > 1 and all(len(chunk.content) <= 300 for chunk in chunks)
    assert all(chunk.content.endswith(".") for chunk in chunks)


def test_model_failure_is_not_swallowed():
    class Unreachable:
        def invoke(self, messages):
            raise ConnectionError("Ollama unreachable")

    with pytest.raises(ConnectionError):
        llm(Unreachable()).chunk([text_element(CATS + STOCKS)], doc_id="d")


# --- factory -------------------------------------------------------------------------------


def test_factory_builds_the_chosen_strategy_with_its_parameters():
    settings = Settings(_env_file=None, table_max_chars=777, embed_batch_size=5)
    embedder, model = TopicEmbedder(), ScriptedModel()

    assert isinstance(get_chunker(None, settings), RecursiveChunker)

    semantic_chunker = get_chunker(
        SemanticChunking(breakpoint_percentile=70), settings, text_embedder=embedder
    )
    assert isinstance(semantic_chunker, SemanticChunker)
    assert (semantic_chunker._percentile, semantic_chunker._batch_size) == (70, 5)
    assert semantic_chunker._embedder is embedder and semantic_chunker._table_max_chars == 777

    llm_chunker = get_chunker(LLMChunking(window_chars=3000), settings, chunking_model=model)
    assert isinstance(llm_chunker, LLMChunker)
    assert llm_chunker._window_chars == 3000 and llm_chunker._model is model
