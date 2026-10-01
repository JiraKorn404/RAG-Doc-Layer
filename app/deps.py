"""The services the UI talks to, created once per server process."""

import threading

import streamlit as st

from ragdoc.services.chat_service import ChatService
from ragdoc.services.document_service import DocumentService


@st.cache_resource(show_spinner=False)
def get_document_service() -> DocumentService:
    service = DocumentService()
    # This runs once, when the server process starts: a document still marked as processing
    # at that point was cut off by a restart. A failure here must not stop the app from loading;
    # the Documents tab reports storage problems itself.
    try:
        service.recover_interrupted()
    except Exception:  # noqa: BLE001
        pass
    return service


@st.cache_resource(show_spinner=False)
def get_chat_service() -> ChatService:
    service = ChatService()
    # Loading the retrieval models takes tens of seconds; start now rather than on the first
    # question. A failure here is not fatal: the first question will try again and report it.
    threading.Thread(target=_warm_up, args=(service,), daemon=True).start()
    return service


def _warm_up(service: ChatService) -> None:
    try:
        service.warm_up()
    except Exception:  # noqa: BLE001
        pass
