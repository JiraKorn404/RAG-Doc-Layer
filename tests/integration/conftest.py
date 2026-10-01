"""Fixtures for tests that need the Docker services (`docker compose up -d`)."""

import importlib.util
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from ragdoc.config import PROJECT_ROOT, Settings, get_settings
from ragdoc.services.document_service import DocumentService
from ragdoc.storage.milvus_store import MilvusStore
from ragdoc.storage.postgres.session import get_engine

TEST_DIM = 8


@pytest.fixture
def db_connection():
    """A connection to the real database inside a transaction that is rolled back afterwards.

    Needs the schema to exist: `uv run alembic upgrade head`.
    """
    with get_engine().connect() as connection:
        transaction = connection.begin()
        try:
            yield connection
        finally:
            transaction.rollback()


@pytest.fixture
def db_session(db_connection):
    session = Session(bind=db_connection, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def scoped_sessions(db_connection):
    """A stand-in for `session_scope` whose commits stay inside the rolled-back transaction."""

    @contextmanager
    def scope():
        session = Session(
            bind=db_connection, join_transaction_mode="create_savepoint", expire_on_commit=False
        )
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return scope


@contextmanager
def temporary_store(**overrides):
    """A store on throwaway collections, dropped afterwards."""
    suffix = uuid.uuid4().hex[:8]
    settings = Settings(
        milvus_uri=get_settings().milvus_uri,
        text_collection=f"test_text_{suffix}",
        image_collection=f"test_image_{suffix}",
        **overrides,
    )
    store = MilvusStore(settings)
    store.ensure_collections()
    try:
        yield settings, store
    finally:
        store.drop_collections()


@pytest.fixture
def milvus_store():
    """A store with small vectors, for tests that supply their own embeddings."""
    with temporary_store(text_embed_dim=TEST_DIM, image_embed_dim=TEST_DIM) as (_settings, store):
        yield store


@pytest.fixture
def make_temporary_store():
    """`temporary_store` for tests that need their own settings (e.g. real embedding sizes)."""
    return temporary_store


def load_sample_builder():
    spec = importlib.util.spec_from_file_location(
        "make_sample_pdf", PROJECT_ROOT / "scripts" / "make_sample_pdf.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_sample_pdf


@pytest.fixture
def sample_pdf(tmp_path) -> Path:
    return load_sample_builder()(tmp_path / "source" / "northwind_report.pdf")


@pytest.fixture
def service_and_store(tmp_path, scoped_sessions, make_temporary_store):
    with make_temporary_store(data_dir=tmp_path / "data") as (settings, store):
        yield DocumentService(settings, store=store, session_scope=scoped_sessions), store, settings
