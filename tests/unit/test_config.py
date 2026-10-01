import pytest
from pydantic import ValidationError

from ragdoc.config import PROJECT_ROOT, Settings


def make_settings(**overrides) -> Settings:
    """Settings built from defaults and overrides only, ignoring any local .env."""
    return Settings(_env_file=None, **overrides)


def test_defaults():
    settings = make_settings()
    assert settings.text_embed_dim == 2560
    assert settings.image_embed_dim == 512
    assert settings.chat_model == settings.vision_model


def test_env_overrides_default(monkeypatch):
    monkeypatch.setenv("TEXT_TOP_K", "9")
    assert make_settings().text_top_k == 9


def test_postgres_dsn():
    settings = make_settings(
        postgres_user="u",
        postgres_password="p",
        postgres_host="h",
        postgres_port=1,
        postgres_db="d",
    )
    assert settings.postgres_dsn == "postgresql+psycopg://u:p@h:1/d"


def test_relative_data_dir_resolves_against_project_root():
    settings = make_settings(data_dir="data")
    assert settings.uploads_dir == PROJECT_ROOT / "data" / "uploads"
    assert settings.images_dir == PROJECT_ROOT / "data" / "images"


def test_absolute_data_dir_is_kept(tmp_path):
    assert make_settings(data_dir=tmp_path).data_path == tmp_path


def test_invalid_value_is_rejected():
    with pytest.raises(ValidationError):
        make_settings(chunk_size=0)
