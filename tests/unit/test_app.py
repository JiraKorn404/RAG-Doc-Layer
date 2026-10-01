"""The Streamlit app run headlessly against fake services."""

import uuid
from datetime import UTC, datetime

import pytest
from streamlit.testing.v1 import AppTest

from ragdoc.config import PROJECT_ROOT
from ragdoc.schemas import (
    ChatEvent,
    ChatEventType,
    ChatResult,
    ChunkType,
    IngestionProgress,
    RetrievalResult,
    RetrievedChunk,
    TraceEvent,
)
from ragdoc.services.chat_service import ConversationInfo, MessageView
from ragdoc.services.document_service import (
    DocumentInfo,
    DuplicateDocumentError,
    IngestionCancelled,
)
from ragdoc.storage.postgres.models import DocumentStatus

APP = PROJECT_ROOT / "app" / "streamlit_app.py"

CHUNKS = [
    RetrievedChunk(
        id="t1",
        doc_id="d",
        chunk_type=ChunkType.TEXT,
        content="## Overview\n\nThe chief executive is Mira Okafor.",
        page=1,
        metadata={"filename": "report.pdf"},
        similarity_score=0.5833,
    ),
    RetrievedChunk(
        id="t2",
        doc_id="d",
        chunk_type=ChunkType.TABLE,
        content="| Year | Revenue |\n| --- | --- |\n| 2024 | $26.3 |",
        page=1,
        metadata={"filename": "report.pdf"},
        similarity_score=0.42,
    ),
    RetrievedChunk(
        id="i1",
        doc_id="d",
        chunk_type=ChunkType.IMAGE,
        content="Figure 1: revenue",
        page=2,
        image_path="images/d/image_001.png",
        metadata={"filename": "report.pdf"},
        similarity_score=0.2838,
    ),
]
TRACE = [
    TraceEvent(node="route", summary="Search the documents: a fact", duration_ms=1200),
    TraceEvent(node="retrieve", summary="Found 2 text/table chunks and 1 images", duration_ms=300),
    TraceEvent(
        node="generate", summary="Answered, citing sources [1]", duration_ms=9000, payload={"k": 1}
    ),
]
ANSWER = "The chief executive is Mira Okafor [1]."


class FakeChatService:
    def __init__(self):
        self.conversations: dict[str, ConversationInfo] = {}
        self.messages: dict[str, list[MessageView]] = {}
        self.fail_with: str | None = None
        self.questions: list[str] = []

    def create_conversation(self, title=None):
        now = datetime.now(UTC)
        info = ConversationInfo(
            id=uuid.uuid4(), title=title or "New conversation", created_at=now, updated_at=now
        )
        self.conversations[str(info.id)] = info
        self.messages[str(info.id)] = []
        return info

    def list_conversations(self):
        return list(self.conversations.values())

    def delete_conversation(self, conversation_id):
        return self.conversations.pop(str(conversation_id), None) is not None

    def get_messages(self, conversation_id):
        return self.messages[str(conversation_id)]

    def image_file(self, chunk):
        return None

    def stream(self, conversation_id, question):
        self.questions.append(question)
        yield ChatEvent(type=ChatEventType.NODE_START, node="route")
        yield ChatEvent(type=ChatEventType.TRACE, node="route", trace=TRACE[0])
        if self.fail_with:
            yield ChatEvent(type=ChatEventType.ERROR, text=self.fail_with)
            return
        yield ChatEvent(type=ChatEventType.REASONING, node="generate", text="Looking at [1]...")
        for piece in ("The chief executive ", "is Mira Okafor [1]."):
            yield ChatEvent(type=ChatEventType.TOKEN, node="generate", text=piece)

        now = datetime.now(UTC)
        conversation_id = str(conversation_id)
        if self.conversations[conversation_id].title == "New conversation":
            self.conversations[conversation_id].title = question
        message_id = uuid.uuid4()
        self.messages[conversation_id] += [
            MessageView(id=uuid.uuid4(), role="user", content=question, created_at=now),
            MessageView(
                id=message_id,
                role="assistant",
                content=ANSWER,
                reasoning="Looking at [1]...",
                created_at=now,
                chunks=CHUNKS,
                used_chunk_ids=["t1"],
                trace=TRACE,
                metrics={"total_ms": 10500, "prompt_tokens": 900, "completion_tokens": 120},
            ),
        ]
        yield ChatEvent(
            type=ChatEventType.FINAL,
            result=ChatResult(
                conversation_id=conversation_id,
                message_id=str(message_id),
                answer=ANSWER,
                retrieval=RetrievalResult(query=question, text=CHUNKS[:2], images=CHUNKS[2:]),
                used_chunk_ids=["t1"],
                trace=TRACE,
            ),
        )


class FakeDocumentService:
    supported_extensions = ["pdf", "txt"]
    ocr_default = False

    def __init__(self):
        self.ocr_seen: list[bool] = []
        self.cancel_next = False
        self.documents: list[DocumentInfo] = [
            self._info(
                "report.pdf",
                n_text_chunks=2,
                n_table_chunks=1,
                n_image_chunks=1,
                ocr_used=True,
                ingest_seconds=83.4,
            ),
            self._info("broken.pdf", status=DocumentStatus.FAILED, error="ValueError: bad file"),
        ]
        self.deleted: list[str] = []

    @staticmethod
    def _info(filename, status=DocumentStatus.READY, **kwargs) -> DocumentInfo:
        return DocumentInfo(
            id=uuid.uuid4(),
            filename=filename,
            status=status,
            created_at=datetime.now(UTC),
            **kwargs,
        )

    def list_documents(self):
        return self.documents

    def ingest(self, filename, data, on_progress=None, ocr=None):
        if any(document.filename == filename for document in self.documents):
            raise DuplicateDocumentError(self.documents[0])
        self.ocr_seen.append(ocr)
        if on_progress:
            on_progress(IngestionProgress(stage="parse", message="Parsing document"))
            on_progress(
                IngestionProgress(
                    stage="parse",
                    message="Parsing pages 5-8 of 9",
                    current=4,
                    total=9,
                    elapsed_s=75,
                )
            )
        if self.cancel_next:
            raise IngestionCancelled("cancelled")
        scanned = filename.startswith("scan")
        document = self._info(
            filename,
            n_text_chunks=0 if scanned else 1,
            n_image_chunks=2 if scanned else 0,
            ocr_used=bool(ocr),
            timings={"parse_ms": 61000, "embed_ms": 9000, "total_ms": 72000},
        )
        self.documents.insert(0, document)
        return document

    def delete(self, document_id):
        self.deleted.append(str(document_id))
        self.documents = [d for d in self.documents if d.id != document_id]
        return True


@pytest.fixture
def app(monkeypatch):
    monkeypatch.syspath_prepend(str(APP.parent))
    import deps

    chat, documents = FakeChatService(), FakeDocumentService()
    monkeypatch.setattr(deps, "get_chat_service", lambda: chat)
    monkeypatch.setattr(deps, "get_document_service", lambda: documents)
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at, chat, documents


def button(at: AppTest, label: str):
    matches = [b for b in at.button if b.label == label]
    assert matches, f"no button {label!r} in {[b.label for b in at.button]}"
    return matches[0]


def all_text(at: AppTest) -> str:
    elements = [*at.markdown, *at.caption, *at.text]
    return "\n".join(str(element.value) for element in elements)


def test_first_render_shows_both_tabs_and_the_documents(app):
    at, _chat, _documents = app
    assert [tab.label for tab in at.tabs] == ["Chat", "Documents"]
    text = all_text(at)
    assert "Ask a question about your uploaded documents" in text
    assert "**report.pdf**" in text and "**broken.pdf**" in text
    assert "Error: ValueError: bad file" in text
    assert ":green-badge[ready]" in text and ":red-badge[failed]" in text


def test_asking_a_question_shows_answer_steps_thinking_and_chunks(app):
    at, chat, _documents = app

    at.chat_input[0].set_value("Who is the chief executive?").run()

    assert not at.exception, at.exception
    assert chat.questions == ["Who is the chief executive?"]
    assert [message.name for message in at.chat_message] == ["user", "assistant"]
    text = all_text(at)
    assert ANSWER in text
    labels = [expander.label for expander in at.expander]
    assert "Thinking and steps (3 steps)" in labels
    assert "Retrieved chunks: 2 text/table, 1 image · 1 cited" in labels
    # Steps, thinking and timings.
    assert "**Route** · 1.2 s" in text and "**Answer** · 9.0 s" in text
    assert "Looking at [1]..." in text
    assert "Total 10.5 s" in text and "tokens in/out 900/120" in text
    # Chunk cards: numbered, with source, page, type, score and the cited badge.
    assert "**[1]** report.pdf · page 1 · text :green-badge[cited in answer]" in text
    assert "**[2]** report.pdf · page 1 · table" in text
    assert "**[3]** report.pdf · page 2 · image" in text
    assert "Similarity score: 0.5833" in text and "Similarity score: 0.2838" in text
    assert "no longer available" in text  # the image file does not exist here
    assert "\\$26.3" in text  # dollar signs are escaped, not rendered as LaTeX
    # The conversation appears in the sidebar under its title.
    assert button(at, "Who is the chief executive?").proto.type == "primary"


def test_follow_up_goes_to_the_same_conversation_and_new_starts_another(app):
    at, chat, _documents = app
    at.chat_input[0].set_value("first").run()
    at.chat_input[0].set_value("second").run()
    assert len(chat.conversations) == 1
    assert len(at.chat_message) == 4

    button(at, "New conversation").click().run()
    assert len(at.chat_message) == 0
    at.chat_input[0].set_value("third").run()
    assert len(chat.conversations) == 2

    button(at, "first").click().run()
    assert len(at.chat_message) == 4


def test_delete_conversation(app):
    at, chat, _documents = app
    at.chat_input[0].set_value("to be deleted").run()
    button(at, "Delete this conversation").click().run()
    assert chat.conversations == {}
    assert len(at.chat_message) == 0


def test_failed_turn_shows_the_error_and_keeps_the_question_visible(app):
    at, chat, _documents = app
    chat.fail_with = "Could not reach Ollama at http://mac:11434."

    at.chat_input[0].set_value("Will this work?").run()

    assert not at.exception, at.exception
    assert "Could not reach Ollama" in at.error[0].value
    assert "Nothing was saved" in at.error[0].value
    assert "Will this work?" in all_text(at)

    # Asking again clears the error.
    chat.fail_with = None
    at.chat_input[0].set_value("Will this work?").run()
    assert len(at.error) == 0


def test_upload_indexes_the_file_and_reports_the_result(app):
    at, _chat, documents = app

    at.file_uploader[0].upload("notes.txt", b"hello", "text/plain").run()
    button(at, "Upload and index").click().run()

    assert not at.exception, at.exception
    assert documents.documents[0].filename == "notes.txt"
    assert at.success[0].value == (
        "**notes.txt**: 1 text, 0 table and 0 image chunks indexed in 1 min 12 s "
        "(parsing 1 min 01 s, embedding 9 s). OCR was off."
    )
    assert documents.ocr_seen == [False]
    assert len(at.warning) == 0
    assert "**notes.txt**" in all_text(at)
    # The uploader is emptied, so the button is disabled again.
    assert button(at, "Upload and index").disabled


def test_ocr_checkbox_is_off_by_default_and_reaches_the_service(app):
    at, _chat, documents = app
    [ocr] = [c for c in at.checkbox if "OCR" in c.label]
    assert ocr.value is False

    ocr.check().run()
    at.file_uploader[0].upload("letter.pdf", b"%PDF", "application/pdf").run()
    button(at, "Upload and index").click().run()

    assert not at.exception, at.exception
    assert documents.ocr_seen == [True]
    assert "OCR was on." in at.success[0].value


def test_document_with_no_text_and_ocr_off_gets_a_warning(app):
    at, _chat, _documents = app
    at.file_uploader[0].upload("scan.pdf", b"%PDF", "application/pdf").run()
    button(at, "Upload and index").click().run()

    assert "0 text, 0 table and 2 image chunks indexed" in at.success[0].value
    assert "no text was found" in at.warning[0].value
    assert "OCR option switched on" in at.warning[0].value


def test_upload_deleted_while_processing_is_reported_as_cancelled(app):
    at, _chat, documents = app
    documents.cancel_next = True
    at.file_uploader[0].upload("big.pdf", b"%PDF", "application/pdf").run()
    button(at, "Upload and index").click().run()

    assert not at.exception, at.exception
    assert "was deleted while it was being processed" in at.info[0].value
    assert len(at.success) == 0 and len(at.error) == 0


def test_documents_table_shows_ocr_and_ingestion_time(app):
    at, _chat, _documents = app
    text = all_text(at)
    assert "OCR" in text and "Time" in text
    values = [str(m.value) for m in at.markdown]
    assert "1 min 23 s" in values and "on" in values and "-" in values


def test_uploading_a_duplicate_is_reported(app):
    at, _chat, documents = app
    at.file_uploader[0].upload("report.pdf", b"%PDF", "application/pdf").run()
    button(at, "Upload and index").click().run()
    assert "already uploaded" in at.warning[0].value
    assert len(documents.documents) == 2


def test_delete_asks_for_confirmation_first(app):
    at, _chat, documents = app
    target = documents.documents[0]

    def by_key(key: str):
        [match] = [b for b in at.button if b.key == key]
        return match

    by_key(f"delete-{target.id}").click().run()
    assert documents.deleted == []  # nothing happens until confirmed
    assert "Everything indexed from it will be removed" in all_text(at)

    by_key(f"cancel-{target.id}").click().run()
    assert documents.deleted == []
    assert not [b for b in at.button if b.key == f"confirm-{target.id}"]

    by_key(f"delete-{target.id}").click().run()
    by_key(f"confirm-{target.id}").click().run()

    assert not at.exception, at.exception
    assert documents.deleted == [str(target.id)]
    assert "**report.pdf** was deleted." in at.success[0].value
    assert [d.filename for d in documents.documents] == ["broken.pdf"]
