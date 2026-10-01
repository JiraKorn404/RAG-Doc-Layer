"""Repositories: all SQL lives here. Each takes a Session; the caller owns the transaction."""

import uuid
from collections.abc import Collection, Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ragdoc.schemas import RetrievedChunk, TraceEvent
from ragdoc.storage.postgres.models import (
    Conversation,
    Document,
    DocumentStatus,
    IngestionMetrics,
    Message,
    QueryMetrics,
    RetrievedChunkRecord,
    TraceEventRecord,
)


class DocumentRepository:
    def __init__(self, session: Session):
        self._session = session

    def create(
        self,
        *,
        filename: str,
        file_hash: str,
        file_path: str,
        document_id: uuid.UUID | None = None,
        chunker: str = "recursive",
        chunk_params: dict[str, Any] | None = None,
    ) -> Document:
        """Register a document as `processing`. Pass `document_id` when the id is needed
        beforehand (it names the upload and image folders)."""
        document = Document(
            id=document_id or uuid.uuid4(),
            filename=filename,
            file_hash=file_hash,
            file_path=file_path,
            chunker=chunker,
            chunk_params=chunk_params,
        )
        self._session.add(document)
        self._session.flush()
        return document

    def get(self, document_id: uuid.UUID) -> Document | None:
        return self._session.get(Document, document_id)

    def get_by_hash(self, file_hash: str) -> Document | None:
        return self._session.scalar(select(Document).where(Document.file_hash == file_hash))

    def list_all(self) -> list[Document]:
        """All documents, newest first."""
        return list(self._session.scalars(select(Document).order_by(Document.created_at.desc())))

    def mark_ready(
        self,
        document_id: uuid.UUID,
        *,
        n_text_chunks: int,
        n_table_chunks: int,
        n_image_chunks: int,
        ocr_used: bool = False,
    ) -> Document:
        document = self._require(document_id)
        document.ocr_used = ocr_used
        document.status = DocumentStatus.READY.value
        document.error = None
        document.n_text_chunks = n_text_chunks
        document.n_table_chunks = n_table_chunks
        document.n_image_chunks = n_image_chunks
        self._session.flush()
        return document

    def list_processing(self) -> list[Document]:
        """Documents whose ingestion has not finished."""
        return list(
            self._session.scalars(
                select(Document).where(Document.status == DocumentStatus.PROCESSING.value)
            )
        )

    def ingestion_seconds(self) -> dict[uuid.UUID, float]:
        """Total ingestion time per document, for those that finished."""
        rows = self._session.execute(
            select(IngestionMetrics.document_id, IngestionMetrics.total_ms)
        )
        return {document_id: total_ms / 1000 for document_id, total_ms in rows}

    def mark_failed(self, document_id: uuid.UUID, error: str) -> Document:
        document = self._require(document_id)
        document.status = DocumentStatus.FAILED.value
        document.error = error
        self._session.flush()
        return document

    def delete(self, document_id: uuid.UUID) -> bool:
        """Delete the registry row (and its metrics). Returns False if it did not exist."""
        document = self.get(document_id)
        if document is None:
            return False
        self._session.delete(document)
        self._session.flush()
        return True

    def add_ingestion_metrics(
        self,
        document_id: uuid.UUID,
        *,
        parse_ms: float,
        chunk_ms: float,
        embed_ms: float,
        store_ms: float,
        total_ms: float,
    ) -> IngestionMetrics:
        metrics = IngestionMetrics(
            document_id=document_id,
            parse_ms=parse_ms,
            chunk_ms=chunk_ms,
            embed_ms=embed_ms,
            store_ms=store_ms,
            total_ms=total_ms,
        )
        self._session.add(metrics)
        self._session.flush()
        return metrics

    def get_ingestion_metrics(self, document_id: uuid.UUID) -> IngestionMetrics | None:
        return self._session.scalar(
            select(IngestionMetrics).where(IngestionMetrics.document_id == document_id)
        )

    def _require(self, document_id: uuid.UUID) -> Document:
        document = self.get(document_id)
        if document is None:
            raise LookupError(f"Document {document_id} not found")
        return document


class ConversationRepository:
    def __init__(self, session: Session):
        self._session = session

    def create(self, title: str | None = None) -> Conversation:
        conversation = Conversation(title=title) if title else Conversation()
        self._session.add(conversation)
        self._session.flush()
        return conversation

    def get(self, conversation_id: uuid.UUID) -> Conversation | None:
        return self._session.get(Conversation, conversation_id)

    def list_all(self) -> list[Conversation]:
        """All conversations, most recently active first."""
        return list(
            self._session.scalars(select(Conversation).order_by(Conversation.updated_at.desc()))
        )

    def rename(self, conversation_id: uuid.UUID, title: str) -> None:
        conversation = self.get(conversation_id)
        if conversation is None:
            raise LookupError(f"Conversation {conversation_id} not found")
        conversation.title = title
        self._session.flush()

    def delete(self, conversation_id: uuid.UUID) -> bool:
        """Delete a conversation and everything recorded under it."""
        conversation = self.get(conversation_id)
        if conversation is None:
            return False
        self._session.delete(conversation)
        self._session.flush()
        return True

    def add_message(
        self,
        conversation_id: uuid.UUID,
        *,
        role: str,
        content: str,
        reasoning: str | None = None,
    ) -> Message:
        conversation = self.get(conversation_id)
        if conversation is None:
            raise LookupError(f"Conversation {conversation_id} not found")
        message = Message(
            conversation_id=conversation_id, role=role, content=content, reasoning=reasoning
        )
        self._session.add(message)
        conversation.updated_at = datetime.now(UTC)
        self._session.flush()
        return message

    def list_messages(self, conversation_id: uuid.UUID, limit: int | None = None) -> list[Message]:
        """Messages in chronological order; with `limit`, only the most recent ones."""
        query = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
        )
        if limit is not None:
            query = query.limit(limit)
        return list(self._session.scalars(query))[::-1]


class TraceRepository:
    """What happened while answering one message: chunks, trace events, metrics."""

    def __init__(self, session: Session):
        self._session = session

    def add_retrieved_chunks(
        self,
        message_id: uuid.UUID,
        chunks: Sequence[RetrievedChunk],
        used_chunk_ids: Collection[str] = (),
    ) -> None:
        """Store chunks in the order given (rank 1 first)."""
        self._session.add_all(
            RetrievedChunkRecord(
                message_id=message_id,
                chunk_id=chunk.id,
                doc_id=chunk.doc_id,
                chunk_type=chunk.chunk_type.value,
                content_snapshot=chunk.content,
                image_path=chunk.image_path,
                page=chunk.page,
                similarity_score=chunk.similarity_score,
                rerank_score=chunk.rerank_score,
                rank=rank,
                used_in_answer=chunk.id in used_chunk_ids,
                chunk_metadata=chunk.metadata,
            )
            for rank, chunk in enumerate(chunks, start=1)
        )
        self._session.flush()

    def list_retrieved_chunks(self, message_id: uuid.UUID) -> list[RetrievedChunkRecord]:
        return list(
            self._session.scalars(
                select(RetrievedChunkRecord)
                .where(RetrievedChunkRecord.message_id == message_id)
                .order_by(RetrievedChunkRecord.rank)
            )
        )

    def add_trace_events(self, message_id: uuid.UUID, events: Sequence[TraceEvent]) -> None:
        self._session.add_all(
            TraceEventRecord(
                message_id=message_id,
                node=event.node,
                summary=event.summary,
                payload=event.payload,
                started_at=event.started_at,
                duration_ms=event.duration_ms,
            )
            for event in events
        )
        self._session.flush()

    def list_trace_events(self, message_id: uuid.UUID) -> list[TraceEventRecord]:
        return list(
            self._session.scalars(
                select(TraceEventRecord)
                .where(TraceEventRecord.message_id == message_id)
                .order_by(TraceEventRecord.started_at)
            )
        )

    def add_query_metrics(self, message_id: uuid.UUID, **values: float | int | str) -> QueryMetrics:
        metrics = QueryMetrics(message_id=message_id, **values)
        self._session.add(metrics)
        self._session.flush()
        return metrics

    def get_query_metrics(self, message_id: uuid.UUID) -> QueryMetrics | None:
        return self._session.scalar(
            select(QueryMetrics).where(QueryMetrics.message_id == message_id)
        )
