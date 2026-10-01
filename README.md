# RAG-Doc-Layer

Transparent, agentic RAG over documents with text, tables and images.

See [CLAUDE.md](CLAUDE.md) for architecture and commands, and [PLAN.md](PLAN.md) for the build plan.

## Quick start

Ollama must be running on the Mac (reachable over Tailscale) with the models named in `.env` pulled.

### In Docker

```powershell
Copy-Item .env.example .env        # then set OLLAMA_BASE_URL to the Mac's Tailscale name
docker compose up -d --build       # Milvus, PostgreSQL and the app
```

Open http://127.0.0.1:8502. The first document upload downloads the parsing and image models
(about 1.1 GB) into a Docker volume; later starts reuse them.

### On the host (development)

```powershell
docker compose up -d milvus postgres
uv sync
uv run alembic upgrade head
uv run python scripts/check_ollama.py
uv run streamlit run app/streamlit_app.py
uv run pytest
```
