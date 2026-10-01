"""Application settings. The only place environment variables are read."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ollama (remote, over Tailscale)
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout_s: float = Field(default=120, gt=0)
    # Seconds, not "30m": OllamaEmbeddings only accepts an integer.
    ollama_keep_alive_s: int = Field(default=1800, ge=0)

    # Models
    chat_model: str = "gemma4:e4b-mlx"
    vision_model: str = "gemma4:e4b-mlx"
    text_embed_model: str = "qwen3-embedding:4b"
    text_embed_dim: int = Field(default=2560, gt=0)
    # Qwen3 embeddings expect a task instruction on queries only, never on documents.
    text_embed_query_instruction: str = (
        "Given a user question, retrieve relevant passages that answer the question"
    )
    image_embed_model: str = "clip-ViT-B-32"
    image_embed_dim: int = Field(default=512, gt=0)

    # Milvus
    milvus_uri: str = "http://127.0.0.1:19530"
    text_collection: str = "text_chunks"
    image_collection: str = "image_chunks"

    # PostgreSQL
    # 127.0.0.1, not localhost: the Docker ports are IPv4-only and localhost tries ::1 first,
    # which hangs on Windows.
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5433
    postgres_user: str = "ragdoc"
    postgres_password: str = "ragdoc"
    postgres_db: str = "ragdoc"

    # Ingestion
    chunker: str = "recursive"
    # Sizes are in characters. The upper bound keeps any chunk inside Milvus's 65535-byte
    # VARCHAR limit even when every character takes 4 bytes.
    chunk_size: int = Field(default=1000, gt=0, le=16000)
    chunk_overlap: int = Field(default=150, ge=0)
    # Tables stay in one chunk up to this size, then are split by rows with the header repeated.
    table_max_chars: int = Field(default=4000, gt=0, le=16000)
    # Extracted pictures smaller than this on either side (icons, bullets, logos) are skipped.
    min_image_side_px: int = Field(default=64, ge=1)
    # Render scale for pictures cut out of PDF pages (1.0 = 72 dpi).
    pdf_image_scale: float = Field(default=2.0, gt=0)
    embed_batch_size: int = Field(default=16, gt=0)
    # Read text inside pictures and scanned pages. Off by default: it makes parsing
    # noticeably slower and digital documents do not need it. Each upload can override it.
    ocr_default: bool = False
    # PDFs are parsed this many pages at a time. Progress is reported, and a cancelled
    # upload stops, between batches.
    parse_page_batch: int = Field(default=4, gt=0)
    # Give up on a document whose parsing takes longer than this.
    parse_timeout_s: float = Field(default=1800, gt=0)

    # Retrieval / agent
    text_top_k: int = Field(default=5, gt=0)
    image_top_k: int = Field(default=3, ge=0)
    max_retries: int = Field(default=2, ge=0)
    history_turns: int = Field(default=6, ge=0)
    # Let the model think before answering, and show that thinking. Slower but more transparent.
    chat_reasoning: bool = True
    # Output limits per model call. Without one, a model that starts repeating itself streams
    # forever, and the request timeout never fires because data keeps arriving.
    control_max_tokens: int = Field(default=300, gt=0)  # routing, query rewriting, grading
    answer_max_tokens: int = Field(default=4096, gt=0)  # the answer, including its thinking
    # Images are downscaled to this longest side before being sent to the model.
    max_image_side_px: int = Field(default=1024, ge=64)

    # Storage on disk
    data_dir: Path = Path("data")

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def data_path(self) -> Path:
        """`data_dir` resolved against the project root when relative."""
        return self.data_dir if self.data_dir.is_absolute() else PROJECT_ROOT / self.data_dir

    @property
    def uploads_dir(self) -> Path:
        return self.data_path / "uploads"

    @property
    def images_dir(self) -> Path:
        return self.data_path / "images"

    def resolve_data_path(self, relative: str) -> Path:
        """Absolute path of a file stored relative to the data directory (e.g. image chunks)."""
        return self.data_path / relative


@lru_cache
def get_settings() -> Settings:
    return Settings()
