# The Streamlit app. Milvus and PostgreSQL are separate services (see docker-compose.yml);
# Ollama runs on the Mac and is reached over Tailscale.

FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# libgl1 and libglib2.0-0: needed by OpenCV, which Docling's OCR uses. curl: the healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# Models are downloaded on first use into /models, which docker-compose mounts as a volume.
# The directories are created here, owned by the app user, so the volume starts out writable.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app /models/hf /models/rapidocr \
    && chown -R app:app /app /models

USER app
WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/models/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DATA_DIR=/app/data \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8502 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

# Dependencies first, in their own layer: code changes do not reinstall them.
COPY --chown=app:app pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/home/app/.cache/uv,uid=1000,gid=1000 \
    uv sync --frozen --no-dev --no-install-project

# RapidOCR keeps its models inside its own package directory. Point that at the volume, so they
# are downloaded once rather than after every container rebuild.
RUN OCR_MODELS="$(python -c 'import rapidocr, pathlib; print(pathlib.Path(rapidocr.__file__).parent / "models")')" \
    && rm -rf "$OCR_MODELS" \
    && ln -s /models/rapidocr "$OCR_MODELS"

COPY --chown=app:app README.md alembic.ini ./
COPY --chown=app:app .streamlit ./.streamlit
COPY --chown=app:app migrations ./migrations
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app src ./src
COPY --chown=app:app app ./app
RUN --mount=type=cache,target=/home/app/.cache/uv,uid=1000,gid=1000 \
    uv sync --frozen --no-dev

EXPOSE 8502

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8502/_stcore/health || exit 1

# Apply database migrations, then start the app.
CMD ["sh", "-c", "alembic upgrade head && exec streamlit run app/streamlit_app.py"]
