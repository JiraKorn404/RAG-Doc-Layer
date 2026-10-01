"""The graph with fake models, retriever and storage: routing, retries, tracing, streaming."""

from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from ragdoc.agent import prompts
from ragdoc.agent.context import cited_chunk_ids, parse_decision, source_blocks, source_label
from ragdoc.agent.graph import build_graph
from ragdoc.agent.nodes.grade import Grade
from ragdoc.agent.nodes.rewrite_query import clean_query
from ragdoc.agent.nodes.route import RouteDecision
from ragdoc.agent.state import AgentDeps, add_usage
from ragdoc.config import Settings
from ragdoc.schemas import ChatEventType, ChunkType, RetrievalResult, RetrievedChunk
from ragdoc.services.chat_service import ChatService, describe_error


class ScriptedModel(BaseChatModel):
    """A chat model that replies from a script.

    `structured` maps a schema class to the objects to return for it, in order. `texts` are the
    plain replies, in order; a reply is streamed word by word so token streaming can be tested.
    """

    structured: dict[Any, list[Any]] = {}
    texts: list[str] = []
    reasoning: str = ""
    seen: list[list[Any]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _next_text(self, messages) -> str:
        self.seen.append(messages)
        # Routing and grading calls are recognised by their system prompt and answered from
        # `structured`, written out in the two-line format those prompts ask for.
        system = messages[0].content
        if system == prompts.ROUTE_SYSTEM:
            decision = self.structured[RouteDecision].pop(0)
            if decision is None:
                return "I am not sure what to do."
            return f"Reason: {decision.reason}\nRoute: {decision.route}"
        if system == prompts.GRADE_SYSTEM:
            grade = self.structured[Grade].pop(0)
            if grade is None:
                return "I am not sure what to do."
            return f"**Reason:** {grade.reason}\n**Relevant:** {'Yes' if grade.relevant else 'No'}."
        return self.texts.pop(0) if self.texts else "ok"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        message = AIMessage(
            self._next_text(messages),
            additional_kwargs={"reasoning_content": self.reasoning} if self.reasoning else {},
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        text = self._next_text(messages)
        if self.reasoning:
            yield ChatGenerationChunk(
                message=AIMessageChunk("", additional_kwargs={"reasoning_content": self.reasoning})
            )
        words = text.split(" ")
        for index, word in enumerate(words):
            piece = word if index == len(words) - 1 else word + " "
            usage = (
                {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
                if index == len(words) - 1
                else None
            )
            yield ChatGenerationChunk(message=AIMessageChunk(piece, usage_metadata=usage))


def chunk(chunk_id: str, chunk_type=ChunkType.TEXT, content="text", **kwargs) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        doc_id="d",
        chunk_type=chunk_type,
        content=content,
        metadata={"filename": "report.pdf"},
        similarity_score=0.5,
        **kwargs,
    )


class FakeRetriever:
    def __init__(self, results: list[RetrievalResult]):
        self.results = results
        self.queries: list[str] = []

    def retrieve(self, query, **kwargs) -> RetrievalResult:
        self.queries.append(query)
        result = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        return result.model_copy(update={"query": query})


class Harness:
    """A graph wired to fakes, with `persist` writing to mocks instead of PostgreSQL."""

    def __init__(self, monkeypatch, *, control, grader, answer, retriever, max_retries=2):
        self.settings = Settings(_env_file=None, max_retries=max_retries)
        self.conversations = MagicMock()
        self.conversations.get.return_value = MagicMock(title="New conversation")
        self.conversations.add_message.return_value = MagicMock(
            id="11111111-1111-1111-1111-111111111111"
        )
        self.conversations.list_messages.return_value = []
        self.traces = MagicMock()
        monkeypatch.setattr(
            "ragdoc.agent.nodes.persist.ConversationRepository", lambda session: self.conversations
        )
        monkeypatch.setattr(
            "ragdoc.agent.nodes.persist.TraceRepository", lambda session: self.traces
        )
        monkeypatch.setattr(
            "ragdoc.services.chat_service.ConversationRepository",
            lambda session: self.conversations,
        )

        @contextmanager
        def scope():
            yield MagicMock()

        self.retriever = retriever
        deps = AgentDeps(
            settings=self.settings,
            control_model=control,
            grader_model=grader,
            answer_model=answer,
            retriever=retriever,
            session_scope=scope,
        )
        self.service = ChatService(self.settings, graph=build_graph(deps), session_scope=scope)

    def run(self, question: str):
        events = list(self.service.stream("22222222-2222-2222-2222-222222222222", question))
        return events, events[-1]


RELEVANT = RetrievalResult(
    query="q",
    text=[chunk("t1", content="Revenue was 26.3"), chunk("t2", ChunkType.TABLE, "| a |")],
    images=[chunk("i1", ChunkType.IMAGE, "Figure 1", image_path="images/d/missing.png")],
)


def nodes_run(events) -> list[str]:
    return [event.trace.node for event in events if event.type is ChatEventType.TRACE]


def test_grounded_answer_path_streams_trace_tokens_and_saves(monkeypatch):
    control = ScriptedModel(
        structured={RouteDecision: [RouteDecision(reason="r", route="retrieve")]}
    )
    grader = ScriptedModel(structured={Grade: [Grade(reason="covers it", relevant=True)]})
    answer = ScriptedModel(texts=["Revenue was 26.3 [1] as charted [3]."], reasoning="thinking...")
    harness = Harness(
        monkeypatch,
        control=control,
        grader=grader,
        answer=answer,
        retriever=FakeRetriever([RELEVANT]),
    )

    events, final = harness.run("What was revenue?")

    assert nodes_run(events) == [
        "route",
        "rewrite_query",
        "retrieve",
        "grade",
        "generate",
        "persist",
    ]
    starts = [event.node for event in events if event.type is ChatEventType.NODE_START]
    assert starts == nodes_run(events)
    tokens = [event.text for event in events if event.type is ChatEventType.TOKEN]
    assert len(tokens) > 1 and "".join(tokens) == "Revenue was 26.3 [1] as charted [3]."
    assert [e.text for e in events if e.type is ChatEventType.REASONING] == ["thinking..."]
    assert all(event.node == "generate" for event in events if event.type is ChatEventType.TOKEN)

    assert final.type is ChatEventType.FINAL
    result = final.result
    assert result.answer == "Revenue was 26.3 [1] as charted [3]."
    assert result.reasoning == "thinking..."
    assert result.used_chunk_ids == ["t1", "i1"]
    assert [c.id for c in result.retrieval.chunks] == ["t1", "t2", "i1"]
    assert result.metrics["n_retries"] == 0
    assert result.metrics["prompt_tokens"] == 10 + 10 + 10
    assert result.message_id == "11111111-1111-1111-1111-111111111111"
    # First question in a conversation needs no model call to form the search query.
    assert harness.retriever.queries == ["What was revenue?"]

    roles = [call.kwargs["role"] for call in harness.conversations.add_message.call_args_list]
    assert roles == ["user", "assistant"]
    harness.conversations.rename.assert_called_once()
    stored_chunks, used = harness.traces.add_retrieved_chunks.call_args.args[1:]
    assert [c.id for c in stored_chunks] == ["t1", "t2", "i1"] and used == ["t1", "i1"]
    stored_trace = harness.traces.add_trace_events.call_args.args[1]
    assert [event.node for event in stored_trace] == [
        "route",
        "rewrite_query",
        "retrieve",
        "grade",
        "generate",
    ]


def test_direct_path_skips_retrieval(monkeypatch):
    retriever = FakeRetriever([RELEVANT])
    harness = Harness(
        monkeypatch,
        control=ScriptedModel(
            structured={RouteDecision: [RouteDecision(reason="hi", route="direct")]}
        ),
        grader=ScriptedModel(),
        answer=ScriptedModel(texts=["Hello there!"]),
        retriever=retriever,
    )

    events, final = harness.run("Hello")

    assert nodes_run(events) == ["route", "direct_answer", "persist"]
    assert final.result.answer == "Hello there!"
    assert final.result.retrieval is None and retriever.queries == []
    harness.traces.add_retrieved_chunks.assert_not_called()


def test_retries_with_new_queries_then_gives_up_without_using_sources(monkeypatch):
    control = ScriptedModel(
        structured={RouteDecision: [RouteDecision(reason="r", route="retrieve")]},
        texts=['"second query"', "third query\nextra line"],
    )
    grader = ScriptedModel(
        structured={Grade: [Grade(reason=f"off topic {i}", relevant=False) for i in range(3)]}
    )
    answer = ScriptedModel(texts=["The documents do not cover this."])
    retriever = FakeRetriever([RELEVANT])
    harness = Harness(
        monkeypatch, control=control, grader=grader, answer=answer, retriever=retriever
    )

    events, final = harness.run("Boiling point of mercury?")

    loop = ["rewrite_query", "retrieve", "grade"]
    assert nodes_run(events) == ["route", *loop, *loop, *loop, "generate", "persist"]
    assert retriever.queries == ["Boiling point of mercury?", "second query", "third query"]
    assert final.result.metrics["n_retries"] == 2
    assert final.result.used_chunk_ids == []
    # The retry prompt lists what was already tried.
    retry_prompt = control.seen[-1][-1].content
    assert "- Boiling point of mercury?" in retry_prompt and "- second query" in retry_prompt
    # The grade only drives the retries: the answer step still gets the sources, and its prompt
    # tells it to say so when they do not cover the question.
    system, *_rest, human = answer.seen[-1]
    assert system.content == prompts.GENERATE_SYSTEM
    assert human.content[0]["text"].startswith("[1] report.pdf")
    generate_event = final.result.trace[-2]
    assert generate_event.payload["graded_relevant"] is False
    assert generate_event.summary == "Found nothing in the sources that answers the question"


def test_relevant_on_retry_stops_the_loop(monkeypatch):
    control = ScriptedModel(
        structured={RouteDecision: [RouteDecision(reason="r", route="retrieve")]},
        texts=["better query"],
    )
    grader = ScriptedModel(
        structured={Grade: [Grade(reason="no", relevant=False), Grade(reason="yes", relevant=True)]}
    )
    harness = Harness(
        monkeypatch,
        control=control,
        grader=grader,
        answer=ScriptedModel(texts=["Answer [2]."]),
        retriever=FakeRetriever([RELEVANT]),
    )
    events, final = harness.run("q")
    assert nodes_run(events).count("retrieve") == 2
    assert final.result.metrics["n_retries"] == 1
    assert final.result.used_chunk_ids == ["t2"]


def test_empty_index_does_not_retry_or_call_the_grader(monkeypatch):
    grader = ScriptedModel()
    harness = Harness(
        monkeypatch,
        control=ScriptedModel(
            structured={RouteDecision: [RouteDecision(reason="r", route="retrieve")]}
        ),
        grader=grader,
        answer=ScriptedModel(texts=["Nothing uploaded yet."]),
        retriever=FakeRetriever([RetrievalResult(query="q")]),
    )
    events, final = harness.run("q")
    assert nodes_run(events) == [
        "route",
        "rewrite_query",
        "retrieve",
        "grade",
        "generate",
        "persist",
    ]
    assert grader.seen == []
    assert final.result.metrics["n_retries"] == 0


def test_unreadable_decisions_fall_back_safely(monkeypatch):
    harness = Harness(
        monkeypatch,
        control=ScriptedModel(structured={RouteDecision: [None]}),
        grader=ScriptedModel(structured={Grade: [None]}),
        answer=ScriptedModel(texts=["Answer [1]."]),
        retriever=FakeRetriever([RELEVANT]),
    )
    events, final = harness.run("q")
    # Unreadable route -> search; unreadable grade -> use the sources.
    assert nodes_run(events) == [
        "route",
        "rewrite_query",
        "retrieve",
        "grade",
        "generate",
        "persist",
    ]
    assert final.result.used_chunk_ids == ["t1"]


def test_history_is_loaded_and_used_to_rewrite_the_query(monkeypatch):
    control = ScriptedModel(
        structured={RouteDecision: [RouteDecision(reason="r", route="retrieve")]},
        texts=["Northwind revenue 2024"],
    )
    retriever = FakeRetriever([RELEVANT])
    harness = Harness(
        monkeypatch,
        control=control,
        grader=ScriptedModel(structured={Grade: [Grade(reason="y", relevant=True)]}),
        answer=ScriptedModel(texts=["26.3 [1]"]),
        retriever=retriever,
    )
    harness.conversations.list_messages.return_value = [
        MagicMock(role="user", content="Tell me about Northwind"),
        MagicMock(role="assistant", content="It builds robots."),
    ]
    harness.conversations.get.return_value = MagicMock(title="Northwind")

    harness.run("And its revenue in 2024?")

    assert retriever.queries == ["Northwind revenue 2024"]
    rewrite_messages = control.seen[-1]
    assert [m.content for m in rewrite_messages[1:]] == [
        "Tell me about Northwind",
        "It builds robots.",
        "And its revenue in 2024?",
    ]
    harness.conversations.rename.assert_not_called()


def test_failure_yields_one_error_event_and_saves_nothing(monkeypatch):
    class BrokenRetriever:
        def retrieve(self, query, **kwargs):
            raise ConnectionError("connection refused")

    harness = Harness(
        monkeypatch,
        control=ScriptedModel(
            structured={RouteDecision: [RouteDecision(reason="r", route="retrieve")] * 2}
        ),
        grader=ScriptedModel(),
        answer=ScriptedModel(),
        retriever=BrokenRetriever(),
    )
    events, last = harness.run("q")
    assert last.type is ChatEventType.ERROR
    assert "Could not reach Ollama" in last.text
    assert not any(event.type is ChatEventType.FINAL for event in events)
    harness.conversations.add_message.assert_not_called()
    with pytest.raises(RuntimeError, match="Ollama"):
        harness.service.ask("22222222-2222-2222-2222-222222222222", "q")


def test_empty_question_is_rejected(monkeypatch):
    harness = Harness(
        monkeypatch,
        control=ScriptedModel(),
        grader=ScriptedModel(),
        answer=ScriptedModel(),
        retriever=FakeRetriever([RELEVANT]),
    )
    [event] = harness.service.stream("22222222-2222-2222-2222-222222222222", "   ")
    assert event.type is ChatEventType.ERROR


# --- helpers ----------------------------------------------------------------------------------


def test_cited_chunk_ids_handles_each_citation_style():
    answer = "A [2]. B [1][3]. C [1, 2]. Out of range [9]. Not a citation [x]."
    assert cited_chunk_ids(answer, RELEVANT) == ["t2", "t1", "i1"]


def test_source_blocks_number_text_then_images(tmp_path):
    from PIL import Image

    settings = Settings(_env_file=None, data_dir=tmp_path)
    (tmp_path / "images").mkdir()
    Image.new("RGB", (2000, 1000), "red").save(tmp_path / "images" / "a.png")
    retrieval = RetrievalResult(
        query="q",
        text=[chunk("t1", content="alpha", page=3)],
        images=[
            chunk("i1", ChunkType.IMAGE, "Figure 1", image_path="images/a.png", page=4),
            chunk("i2", ChunkType.IMAGE, "", image_path="images/gone.png"),
        ],
    )
    blocks = source_blocks(retrieval, settings)

    assert [block["type"] for block in blocks] == ["text", "text", "image_url", "text"]
    assert blocks[0]["text"] == "[1] report.pdf, page 3, text\nalpha"
    assert blocks[1]["text"].startswith("[2] report.pdf, page 4, image\nSource [2] is the image")
    assert "Figure 1" in blocks[1]["text"]
    assert blocks[2]["image_url"].startswith("data:image/png;base64,")
    assert "no longer available" in blocks[3]["text"]
    assert source_label(3, retrieval.images[1]) == "[3] report.pdf, image"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Reason: It asks for a fact.\nRoute: retrieve", ("retrieve", "It asks for a fact.")),
        ("**Reason:** Small talk.\n**Route:** `Direct`.", ("direct", "Small talk.")),
        ("reason: two\nlines of it\nroute: direct\n", ("direct", "two\nlines of it")),
        ("Route: retrieve", ("retrieve", "")),
        # The last decision line wins; an unknown value or a missing line is no decision.
        (
            "Route: direct\nReason: changed my mind\nRoute: retrieve",
            ("retrieve", "changed my mind"),
        ),
        ("Reason: unsure\nRoute: maybe", (None, "unsure")),
        ("I think we should search.", (None, "")),
    ],
)
def test_parse_decision(text, expected):
    assert parse_decision(text, "Route", ("retrieve", "direct")) == expected


def test_clean_query():
    assert clean_query('"revenue 2024"\nExplanation: ...', "fallback") == "revenue 2024"
    assert clean_query("   \n  ", "fallback") == "fallback"


def test_add_usage_sums_counts():
    assert add_usage({"prompt_tokens": 1}, {"prompt_tokens": 2, "completion_tokens": 3}) == {
        "prompt_tokens": 3,
        "completion_tokens": 3,
    }


def test_describe_error_falls_back_to_the_exception_text():
    settings = Settings(_env_file=None)
    assert describe_error(ValueError("boom"), settings) == "ValueError: boom"
    try:
        try:
            raise ConnectionError("refused")
        except ConnectionError as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as outer:
        assert "Could not reach Ollama" in describe_error(outer, settings)
