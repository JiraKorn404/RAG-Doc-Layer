"""RAG-Doc-Layer UI. Run: uv run streamlit run app/streamlit_app.py"""

import streamlit as st

import deps
from tabs import chat, documents

st.set_page_config(page_title="RAG-Doc-Layer", page_icon=":material/find_in_page:", layout="wide")
st.title("RAG-Doc-Layer")

chat_tab, documents_tab = st.tabs(["Chat", "Documents"])
with chat_tab:
    chat.render(deps.get_chat_service())
with documents_tab:
    documents.render(deps.get_document_service())
