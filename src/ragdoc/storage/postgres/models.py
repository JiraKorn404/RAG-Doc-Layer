"""SQLAlchemy models. Schema changes go through an Alembic migration."""

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class DocumentStatus(StrEnum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Base(DeclarativeBase):
    pass


def _pk() -> Mapped[uuid.UUID]:
    return mapped_column(Uuid, primary_key=True, default=uuid.uuid4)


def _now() -> datetime:
    return datetime.now(UTC)


def _created_at() -> Mapped[datetime]:
    # Set in Python, not only by the server: now() in PostgreSQL is the transaction start time,
    # so rows written in one transaction would otherwise share a timestamp and lose their order.
    return mapped_column(DateTime(timezone=True), default=_now, server_default=func.now())


def _message_fk(**kwargs: Any) -> Mapped[uuid.UUID]:
    return mapped_column(ForeignKey("messages.id", ondelete="CASCADE"), index=True, **kwargs)


class Document(Base):
    """Registry of uploaded documents. `id` is the `doc_id` stored on Milvus chunks."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = _pk()
    filename: Mapped[str] = mapped_column(String(512))
    file_hash: Mapped[str] = mapped_column(String(64), unique=True)
    file_path: Mapped[str] = mapped_column(String(1024))
    status: Mapped[str] = mapped_column(String(16), default=DocumentStatus.PROCESSING.value)
    error: Mapped[str | None] = mapped_column(Text)
    n_text_chunks: Mapped[int] = mapped_column(Integer, default=0)
    n_table_chunks: Mapped[int] = mapped_column(Integer, default=0)
    n_image_chunks: Mapped[int] = mapped_column(Integer, default=0)
    # Whether OCR was switched on for this upload.
    ocr_used: Mapped[bool] = mapped_column(default=False, server_default="false")
    # The chunking strategy chosen for this upload, and its parameters.
    chunker: Mapped[str] = mapped_column(
        String(32), default="recursive", server_default="recursive"
    )
    chunk_params: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = _created_at()


class IngestionMetrics(Base):
    __tablename__ = "ingestion_metrics"

    id: Mapped[uuid.UUID] = _pk()
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), unique=True
    )
    parse_ms: Mapped[float] = mapped_column(Float, default=0)
    chunk_ms: Mapped[float] = mapped_column(Float, default=0)
    embed_ms: Mapped[float] = mapped_column(Float, default=0)
    store_ms: Mapped[float] = mapped_column(Float, default=0)
    total_ms: Mapped[float] = mapped_column(Float, default=0)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = _pk()
    title: Mapped[str] = mapped_column(String(255), default="New conversation")
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now, server_default=func.now()
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = _pk()
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    reasoning: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()


class RetrievedChunkRecord(Base):
    """A chunk retrieved for an answer. Not a foreign key to the document: the snapshot must
    stay readable after the document is deleted."""

    __tablename__ = "retrieved_chunks"

    id: Mapped[uuid.UUID] = _pk()
    message_id: Mapped[uuid.UUID] = _message_fk()
    chunk_id: Mapped[str] = mapped_column(String(64))
    doc_id: Mapped[str] = mapped_column(String(64))
    chunk_type: Mapped[str] = mapped_column(String(16))
    content_snapshot: Mapped[str] = mapped_column(Text)
    image_path: Mapped[str | None] = mapped_column(String(1024))
    page: Mapped[int | None] = mapped_column(Integer)
    similarity_score: Mapped[float] = mapped_column(Float)
    rerank_score: Mapped[float | None] = mapped_column(Float)
    # Position in the list shown to the model: the number it cites as [rank].
    rank: Mapped[int] = mapped_column(Integer)
    used_in_answer: Mapped[bool] = mapped_column(default=False)
    # The column is "metadata"; the attribute cannot be, since SQLAlchemy reserves that name.
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict, server_default="{}"
    )


class TraceEventRecord(Base):
    __tablename__ = "trace_events"

    id: Mapped[uuid.UUID] = _pk()
    message_id: Mapped[uuid.UUID] = _message_fk()
    node: Mapped[str] = mapped_column(String(64))
    summary: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[float] = mapped_column(Float, default=0)


class QueryMetrics(Base):
    __tablename__ = "query_metrics"

    id: Mapped[uuid.UUID] = _pk()
    message_id: Mapped[uuid.UUID] = _message_fk(unique=True)
    total_ms: Mapped[float] = mapped_column(Float, default=0)
    retrieve_ms: Mapped[float] = mapped_column(Float, default=0)
    grade_ms: Mapped[float] = mapped_column(Float, default=0)
    generate_ms: Mapped[float] = mapped_column(Float, default=0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    n_retries: Mapped[int] = mapped_column(Integer, default=0)
    chat_model: Mapped[str] = mapped_column(String(128), default="")
    embed_model: Mapped[str] = mapped_column(String(128), default="")
