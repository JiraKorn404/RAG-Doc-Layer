"""Tab 1: the RAG conversation, with the agent's steps, thinking and retrieved chunks."""

import streamlit as st

from components.chunks import render_sources
from components.text import safe_markdown
from components.trace import render_process, render_steps, running_label
from ragdoc.schemas import ChatEventType
from ragdoc.services.chat_service import ChatService, MessageView

CONVERSATION = "conversation_id"  # session key: the open conversation, or None for a new one
LAST_ERROR = "chat_error"  # session key: (question, message) of a turn that failed


def render_sidebar(service: ChatService) -> None:
    with st.sidebar:
        st.header("Conversations")
        if st.button("New conversation", icon=":material/add:", width="stretch"):
            st.session_state[CONVERSATION] = None
            st.session_state.pop(LAST_ERROR, None)
            st.rerun()

        try:
            conversations = service.list_conversations()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not load conversations: {exc}")
            return

        current = st.session_state.get(CONVERSATION)
        for conversation in conversations:
            is_current = str(conversation.id) == current
            if st.button(
                conversation.title,
                key=f"conversation-{conversation.id}",
                type="primary" if is_current else "secondary",
                width="stretch",
            ):
                st.session_state[CONVERSATION] = str(conversation.id)
                st.session_state.pop(LAST_ERROR, None)
                st.rerun()

        if current and any(str(c.id) == current for c in conversations):
            st.divider()
            if st.button("Delete this conversation", icon=":material/delete:", width="stretch"):
                service.delete_conversation(current)
                st.session_state[CONVERSATION] = None
                st.rerun()


def render_message(service: ChatService, message: MessageView) -> None:
    with st.chat_message(message.role):
        if message.role != "assistant":
            st.markdown(safe_markdown(message.content))
            return
        render_process(message.trace, message.reasoning, message.metrics, key=str(message.id))
        st.markdown(safe_markdown(message.content))
        render_sources(message.chunks, message.used_chunk_ids, service.image_file)


def answer(service: ChatService, conversation_id: str, question: str) -> bool:
    """Run one turn, showing the agent's work live. Returns True if the turn was saved."""
    with st.chat_message("user"):
        st.markdown(safe_markdown(question))

    with st.chat_message("assistant"):
        # Not `with status:` - leaving that block marks the status complete and collapses it.
        status = st.status("Starting...", expanded=True)
        steps = status.container()
        thinking_title = status.empty()
        thinking_box = status.empty()
        answer_box = st.empty()

        reasoning = text = ""
        for event in service.stream(conversation_id, question):
            if event.type is ChatEventType.NODE_START:
                # `expanded` must be repeated: an update without it collapses the status.
                status.update(label=running_label(event.node), expanded=True)
            elif event.type is ChatEventType.TRACE:
                with steps:
                    render_steps([event.trace])
            elif event.type is ChatEventType.REASONING:
                reasoning += event.text
                thinking_title.markdown("**Model thinking**")
                thinking_box.markdown(safe_markdown(reasoning))
            elif event.type is ChatEventType.TOKEN:
                text += event.text
                answer_box.markdown(safe_markdown(text) + " ▌")
            elif event.type is ChatEventType.ERROR:
                status.update(label="Failed", state="error", expanded=True)
                st.session_state[LAST_ERROR] = (question, event.text)
                return False
            elif event.type is ChatEventType.FINAL:
                status.update(label="Done", state="complete", expanded=False)
                answer_box.markdown(safe_markdown(event.result.answer))
    return True


def render(service: ChatService) -> None:
    render_sidebar(service)
    conversation_id = st.session_state.get(CONVERSATION)

    if conversation_id:
        try:
            messages = service.get_messages(conversation_id)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not load this conversation: {exc}")
            messages = []
        for message in messages:
            render_message(service, message)
    else:
        st.caption(
            "Ask a question about your uploaded documents. Each answer shows the steps the "
            "assistant took, its thinking, and the chunks it retrieved with their scores."
        )

    # Inside tabs the chat input is placed where it is called, so the live turn gets a container
    # declared before it: the new question and answer then appear above the input, like history.
    live = st.container()
    question = st.chat_input("Ask about your documents")

    if not question:
        if error := st.session_state.get(LAST_ERROR):
            failed_question, message = error
            with live:
                with st.chat_message("user"):
                    st.markdown(safe_markdown(failed_question))
                st.error(f"{message}\n\nNothing was saved for this question. You can ask it again.")
        return

    st.session_state.pop(LAST_ERROR, None)
    is_new = not conversation_id
    if is_new:
        conversation_id = str(service.create_conversation().id)
        st.session_state[CONVERSATION] = conversation_id
    with live:
        saved = answer(service, conversation_id, question)
    if not saved and is_new:
        # Do not leave an empty conversation behind in the sidebar.
        service.delete_conversation(conversation_id)
        st.session_state[CONVERSATION] = None
    # Redraw from what was saved, so a live answer and a reloaded one look the same.
    st.rerun()
