"""The real app, headless, against the real services: upload, ask, delete.

Unlike the other integration tests this one cannot be isolated in a transaction, because the
app builds its own services. It uploads a uniquely named file into the real stores and removes
what it created afterwards.
"""

import uuid

import pytest
from streamlit.testing.v1 import AppTest

from ragdoc.config import PROJECT_ROOT
from ragdoc.services.chat_service import ChatService
from ragdoc.services.document_service import DocumentService

pytestmark = pytest.mark.integration

APP = PROJECT_ROOT / "app" / "streamlit_app.py"


def button(at: AppTest, *, label: str | None = None, key: str | None = None):
    matches = [b for b in at.button if (label and b.label == label) or (key and b.key == key)]
    assert matches, f"no button {label or key!r} in {[(b.label, b.key) for b in at.button]}"
    return matches[0]


def all_text(at: AppTest) -> str:
    return "\n".join(str(e.value) for e in [*at.markdown, *at.caption, *at.text])


def test_upload_ask_delete(monkeypatch):
    monkeypatch.syspath_prepend(str(APP.parent))
    token = uuid.uuid4().hex[:8]
    filename = f"e2e_{token}.txt"
    content = (
        f"Project Zephyr-{token} status note.\n\n"
        f"The launch code name for project Zephyr-{token} is Blue Heron. "
        "The launch is planned for 14 March 2027 from the Lisbon office."
    ).encode()
    documents, chat = DocumentService(), ChatService()
    before = {str(c.id) for c in chat.list_conversations()}

    try:
        at = AppTest.from_file(str(APP), default_timeout=600)
        at.run()
        assert not at.exception, at.exception

        # Upload
        at.file_uploader[0].upload(filename, content, "text/plain").run()
        button(at, label="Upload and index").click().run()
        assert not at.exception, at.exception
        assert at.success and filename in at.success[0].value, [e.value for e in at.error]
        assert f"**{filename}**" in all_text(at)
        [document] = [d for d in documents.list_documents() if d.filename == filename]

        # Ask
        at.chat_input[0].set_value(f"What is the launch code name for project Zephyr-{token}?")
        at.run()
        assert not at.exception, at.exception
        assert not at.error, [e.value for e in at.error]
        assert [message.name for message in at.chat_message] == ["user", "assistant"]
        text = all_text(at)
        assert "Blue Heron" in text
        assert any(e.label.startswith("Thinking and steps") for e in at.expander)
        assert any(e.label.startswith("Retrieved chunks") for e in at.expander)
        assert f"{filename}" in text and "Similarity score:" in text

        # Delete
        button(at, key=f"delete-{document.id}").click().run()
        button(at, key=f"confirm-{document.id}").click().run()
        assert not at.exception, at.exception
        assert filename not in [d.filename for d in documents.list_documents()]
        assert set(documents.store.count_chunks(str(document.id)).values()) == {0}
    finally:
        for document in documents.list_documents():
            if document.filename == filename:
                documents.delete(document.id)
        for conversation in chat.list_conversations():
            if str(conversation.id) not in before:
                chat.delete_conversation(conversation.id)
