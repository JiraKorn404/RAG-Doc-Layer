"""Tab 2: upload documents, see what is indexed, delete."""

import streamlit as st
from pydantic import ValidationError

from components.text import duration
from ragdoc.schemas import ChunkingConfig, IngestionProgress
from ragdoc.services.document_service import (
    DocumentInfo,
    DocumentService,
    DuplicateDocumentError,
    IngestionCancelled,
    UnsupportedFileTypeError,
)

UPLOADER_VERSION = "uploader_version"  # session key: bumped to clear the uploader
RESULTS = "upload_results"  # session key: [(kind, message)] from the last upload or delete
PENDING_DELETE = "pending_delete"  # session key: id of the document awaiting confirmation

STATUS_BADGES = {
    "ready": ":green-badge[ready]",
    "processing": ":orange-badge[processing]",
    "failed": ":red-badge[failed]",
}


def summary(document: DocumentInfo) -> list[tuple[str, str]]:
    """The result messages for a document that was indexed."""
    timings = document.timings
    took = (
        f" in {duration(timings['total_ms'] / 1000)} "
        f"(parsing {duration(timings.get('parse_ms', 0) / 1000)}, "
        f"embedding {duration(timings.get('embed_ms', 0) / 1000)})"
        if timings.get("total_ms") is not None
        else ""
    )
    messages = [
        (
            "success",
            f"**{document.filename}**: {document.n_text_chunks} text, "
            f"{document.n_table_chunks} table and {document.n_image_chunks} image chunks "
            f"indexed{took}. OCR was {'on' if document.ocr_used else 'off'}. "
            f"Chunking: {document.chunker}.",
        )
    ]
    if document.chunk_fallbacks:
        messages.append(
            (
                "warning",
                f"**{document.filename}**: the model's reply could not be used for "
                f"{document.chunk_fallbacks} part(s) of the text, which were split by size "
                "instead.",
            )
        )
    if not document.ocr_used and document.n_text_chunks + document.n_table_chunks == 0:
        messages.append(
            (
                "warning",
                f"**{document.filename}**: no text was found, so only its pictures can be "
                "searched. If it is a scanned document, delete it and upload it again with the "
                "OCR option switched on.",
            )
        )
    return messages


def chunking_inputs(service: DocumentService) -> ChunkingConfig | None:
    """The strategy picker and the parameters of the chosen strategy.

    Returns None, after showing why, when the values entered do not fit together. The inputs
    are generated from the strategy's model in `schemas.py`: label, help text and range of each
    parameter come from its field.
    """
    defaults = service.chunking_defaults
    names = list(defaults)
    strategy = st.selectbox(
        "Chunking strategy",
        names,
        index=names.index(service.default_chunker),
        format_func=lambda name: defaults[name].label,
        help=(
            "How the text of the documents is cut into searchable pieces. Tables and images "
            "are handled the same way whichever you choose."
        ),
    )
    default = defaults[strategy]
    st.caption(default.summary)

    values: dict[str, int] = {}
    with st.expander("Parameters"):
        for name, field in type(default).model_fields.items():
            if name == "strategy":
                continue
            low = next((m.ge for m in field.metadata if hasattr(m, "ge")), None)
            high = next((m.le for m in field.metadata if hasattr(m, "le")), None)
            # Keyed by strategy, so each strategy keeps the values entered for it.
            values[name] = st.number_input(
                field.title or name,
                min_value=low,
                max_value=high,
                value=getattr(default, name),
                step=1 if high is not None and high <= 100 else 50,
                help=field.description,
                key=f"chunking-{strategy}-{name}",
            )
    try:
        return type(default)(**values)
    except ValidationError as exc:
        st.error("; ".join(error["msg"].removeprefix("Value error, ") for error in exc.errors()))
        return None


def ingest_files(service: DocumentService, files, ocr: bool, chunking: ChunkingConfig) -> None:
    results: list[tuple[str, str]] = []
    for file in files:
        # Not `with st.status(...)`: leaving that block would mark the status complete.
        status = st.status(f"Processing {file.name}...", expanded=True)
        bar = status.progress(0.0)
        line = status.empty()

        def show(progress: IngestionProgress, bar=bar, line=line) -> None:
            if progress.fraction is not None:
                bar.progress(progress.fraction)
            counts = (
                f" ({progress.current} of {progress.total} done)"
                if progress.total and progress.stage != "parse"
                else ""
            )
            line.markdown(f"{progress.message}{counts} · {duration(progress.elapsed_s)} elapsed")

        try:
            document = service.ingest(
                file.name, file.getvalue(), on_progress=show, ocr=ocr, chunking=chunking
            )
        except DuplicateDocumentError as exc:
            status.update(label=f"{file.name}: already uploaded", state="error")
            results.append(("warning", f"**{file.name}**: {exc}"))
            continue
        except UnsupportedFileTypeError as exc:
            status.update(label=f"{file.name}: not supported", state="error")
            results.append(("error", f"**{file.name}**: {exc}"))
            continue
        except IngestionCancelled:
            status.update(label=f"{file.name}: cancelled", state="error")
            results.append(("info", f"**{file.name}** was deleted while it was being processed."))
            continue
        except Exception as exc:  # noqa: BLE001
            status.update(label=f"{file.name}: failed", state="error")
            results.append(("error", f"**{file.name}**: {type(exc).__name__}: {exc}"))
            continue

        if document.status.value == "ready":
            bar.progress(1.0)
            status.update(label=f"{file.name}: done", state="complete", expanded=False)
            results.extend(summary(document))
        else:
            status.update(label=f"{file.name}: failed", state="error")
            results.append(("error", f"**{file.name}** could not be processed: {document.error}"))
    st.session_state[RESULTS] = results


def render_upload(service: DocumentService) -> None:
    st.subheader("Upload")
    version = st.session_state.setdefault(UPLOADER_VERSION, 0)
    files = st.file_uploader(
        "Add documents",
        type=service.supported_extensions,
        accept_multiple_files=True,
        key=f"uploader-{version}",
        help="Text, tables and images inside each document are indexed separately.",
    )
    ocr = st.checkbox(
        "Read text inside images and scanned pages (OCR, slower)",
        value=service.ocr_default,
        help=(
            "Leave off for normal digital documents. Switch on for scanned documents, or when "
            "text that only appears inside pictures (chart labels, screenshots) should be "
            "searchable. It is never switched on automatically."
        ),
    )
    chunking = chunking_inputs(service)
    if st.button("Upload and index", type="primary", disabled=not files or chunking is None):
        ingest_files(service, files, ocr, chunking)
        # A new key gives an empty uploader, so the same files are not offered again.
        st.session_state[UPLOADER_VERSION] = version + 1
        st.rerun()

    for kind, message in st.session_state.pop(RESULTS, []):
        getattr(st, kind)(message)


def render_confirmation(service: DocumentService, document: DocumentInfo) -> None:
    """The confirmation shown under a row after its Delete button was pressed."""
    with st.container(border=True):
        st.write(
            f"Delete **{document.filename}**? Everything indexed from it will be removed. Past "
            "answers keep the text they quoted, but their images will no longer be shown."
        )
        confirm, cancel, _rest = st.columns([1, 1, 4])
        if confirm.button(
            "Yes, delete", key=f"confirm-{document.id}", type="primary", width="stretch"
        ):
            st.session_state.pop(PENDING_DELETE, None)
            try:
                service.delete(document.id)
            except Exception as exc:  # noqa: BLE001
                st.session_state[RESULTS] = [
                    ("error", f"Could not delete **{document.filename}**: {exc}")
                ]
            else:
                st.session_state[RESULTS] = [("success", f"**{document.filename}** was deleted.")]
            st.rerun()
        if cancel.button("Cancel", key=f"cancel-{document.id}", width="stretch"):
            st.session_state.pop(PENDING_DELETE, None)
            st.rerun()


def render_list(service: DocumentService) -> None:
    title, refresh = st.columns([5, 1], vertical_alignment="bottom")
    title.subheader("Documents")
    if refresh.button("Refresh", icon=":material/refresh:", width="stretch"):
        st.rerun()

    try:
        documents = service.list_documents()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load documents: {exc}")
        return
    if not documents:
        st.info("No documents yet. Upload one above.")
        return

    widths = [3.2, 1.3, 0.7, 0.8, 0.8, 0.6, 1.3, 1.1, 1.8, 1.2]
    labels = "Name Status Text Tables Images OCR Chunking Time Uploaded".split() + [""]
    strategies = service.chunking_defaults
    header = st.columns(widths)
    for column, label in zip(header, labels, strict=True):
        column.caption(label)

    for document in documents:
        row = st.columns(widths, vertical_alignment="center")
        row[0].markdown(f"**{document.filename}**")
        row[1].markdown(STATUS_BADGES.get(document.status.value, document.status.value))
        row[2].write(document.n_text_chunks)
        row[3].write(document.n_table_chunks)
        row[4].write(document.n_image_chunks)
        row[5].write("on" if document.ocr_used else "off")
        strategy = strategies.get(document.chunker)
        params = ", ".join(f"{k} = {v}" for k, v in (document.chunk_params or {}).items())
        row[6].markdown(strategy.label if strategy else document.chunker, help=params or None)
        took = document.ingest_seconds
        row[7].write(duration(took) if took is not None else "-")
        row[8].write(document.created_at.astimezone().strftime("%Y-%m-%d %H:%M"))
        if row[9].button("Delete", key=f"delete-{document.id}", icon=":material/delete:"):
            st.session_state[PENDING_DELETE] = str(document.id)
            st.rerun()
        if document.error:
            st.caption(f"Error: {document.error}")
        if st.session_state.get(PENDING_DELETE) == str(document.id):
            render_confirmation(service, document)


def render(service: DocumentService) -> None:
    render_upload(service)
    st.divider()
    render_list(service)
