# RAG-Doc-Layer

Transparent, agentic RAG over documents that contain text, tables and images.
Self-hosted across two machines: the app, Milvus and PostgreSQL run in Docker on the Windows dev
machine; Ollama runs on a MacBook M1 Pro (32 GB) reached over Tailscale. Single user, no auth.
The app can also be run directly on the host with `uv`, which is how it is developed and tested.

See [PLAN.md](PLAN.md) for the build phases, data schemas and open questions. Keep it in sync when a
decision changes.

## Stack

| Concern | Choice |
|---|---|
| Language / tooling | Python 3.12, `uv`, `ruff`, `pytest` |
| Chat, vision | `langchain-ollama` `ChatOllama`, model `gemma4:e4b-mlx` |
| Text embeddings | `langchain-ollama` `OllamaEmbeddings`, model `qwen3-embedding:4b` (2560-dim) |
| Image embeddings | `sentence-transformers` `clip-ViT-B-32` (512-dim), runs in the app process on the Windows machine because Ollama cannot embed images |
| Orchestration | LangGraph custom `StateGraph` (not a prebuilt ReAct agent) |
| Parsing | Docling (text, tables as Markdown, extracted figures, OCR) |
| Chunking | Recursive (`RecursiveCharacterTextSplitter`); tables kept whole |
| Vector DB | Milvus standalone (Docker), accessed through `pymilvus` `MilvusClient` |
| Relational DB | PostgreSQL (Docker), SQLAlchemy 2.x + Alembic |
| Reranking | Not implemented yet. Planned as a later phase (see PLAN.md); retrieval is vector search only for now |
| Frontend | Streamlit, two tabs: Chat and Documents |

## Architecture

```
Streamlit (app/)  ->  services/  ->  agent/ (LangGraph)  ->  retrieval/  ->  storage/
                                 ->  ingestion/          ->  llm/        ->  storage/
```

Dependencies point one way, left to right. Lower layers never import from higher ones.

- **`app/`** only renders and calls `services/`. No pymilvus, SQLAlchemy, or LangChain imports here;
  from `ragdoc` it imports only `services` and `schemas`. Streamlit puts `app/` on the import
  path, so its modules import each other as `deps`, `tabs.chat`, `components.trace`.
- **`services/`** is the single entry point for the UI (`ChatService`, `DocumentService`). A FastAPI
  layer could be added later on top of it without touching anything else.
- **`agent/`** holds the graph. One file per node; nodes are plain functions `(state) -> partial state`.
- **`retrieval/`**, **`ingestion/`**, **`llm/`**, **`storage/`** each expose a small interface
  (`Protocol`/ABC) plus implementations, built through a factory that reads `Settings`.

### Query graph

```
route ─┬─> direct_answer ──────────────────────────────> persist
       └─> rewrite_query -> retrieve -> grade ─┬─> generate -> persist
                 ^                              │
                 └────── (context not relevant, retries left)
```

- `retrieve` searches two Milvus collections: `text_chunks` (text + tables, Ollama text embedding)
  and `image_chunks` (CLIP image embedding, queried with the CLIP text encoder).
  It returns the top `TEXT_TOP_K` and top `IMAGE_TOP_K` as two separate lists.
- `grade` is one call to the chat model: do the sources help with any part of the question? If
  not and retries remain, the query is rewritten and retrieval runs again. The grade only drives
  the retries: the grader is sometimes wrong (notably about what an image shows).
- `generate` gets both lists as numbered sources (text and tables first, then images, which are
  attached as images) and must cite them as `[n]`. Cited chunks are saved as `used_in_answer`.
  It always gets the last retrieved sources, whatever the grade, and its prompt tells it to say
  so when they do not cover the question. Only when nothing is indexed does it get no sources.
- `rewrite_query` makes no model call for the first question of a conversation. With history it
  makes the question standalone; on a retry it writes a query different from those already tried.
- Three model clients (`AgentDeps`): `control_model` (routing, rewriting) and `grader_model` run
  without thinking and with a short output limit; `answer_model` thinks if `CHAT_REASONING` is on.
- A `rerank` node will later sit between `retrieve` and `grade`. Until then, do not add reranking
  logic anywhere else; `RetrievedChunk.rerank_score` exists but is always `None`.

### Transparency

Nodes are plain functions returning a `NodeResult` (state update, one-line summary, payload).
`traced()` in `agent/graph.py` wraps each one: it announces the start, times the node and appends
a `TraceEvent` to the state. `ChatService.stream()` turns a graph run into `ChatEvent`s:
`NODE_START` and `TRACE` around every step, `REASONING` and `TOKEN` pieces while the answer is
written (only from the answer nodes), then one `FINAL` with the saved `ChatResult`, or one `ERROR`.
The UI shows, per answer: the step-by-step trace, the model's thinking, and each retrieved chunk
with its source document, page, type, similarity score and whether it was cited. The same trace is
saved to PostgreSQL, and `ChatService.get_messages()` returns it for past answers. A failed turn
saves nothing.

## Layout

```
app/
  streamlit_app.py          # entry point, tab wiring only
  deps.py                   # the two services, cached per server process
  tabs/chat.py              # tab 1 (also renders the conversation sidebar)
  tabs/documents.py         # tab 2
  components/               # trace.py (steps, thinking, metrics), chunks.py (chunk cards), text.py
.streamlit/config.toml      # port 8502, bound to 127.0.0.1
src/ragdoc/
  config.py                 # pydantic-settings `Settings`, the only place env vars are read
  llm/                      # chat.py, embeddings.py, image_embeddings.py
  ingestion/
    parser.py               # Docling -> list[ParsedElement] (text | table | image)
    chunking/               # base.py (Chunker protocol), recursive.py, factory.py
    pipeline.py             # parse -> chunk -> embed -> store
  storage/
    milvus_store.py         # collections, insert, search, delete by doc_id
    postgres/               # models.py, session.py, repositories.py
  retrieval/                # retriever.py (reranker.py arrives with the reranking phase)
  agent/                    # state.py, graph.py, prompts.py, nodes/
  services/                 # chat_service.py, document_service.py
  schemas.py                # shared pydantic models: Chunk, RetrievedChunk, TraceEvent
migrations/                 # Alembic (env.py takes the URL from Settings); alembic.ini at the root
tests/                      # unit/ and integration/
data/                       # uploads/, images/ (gitignored)
scripts/                    # check_ollama.py, documents.py, make_sample_pdf.py, search.py, chat.py
docker/milvus/              # embedded-etcd config mounted into the Milvus container
docker-compose.yml          # milvus standalone (embedded etcd, local storage), postgres, app
Dockerfile, .dockerignore   # the app image
```

## Commands

```powershell
# Everything in Docker: app at http://127.0.0.1:8502
docker compose up -d                      # Milvus, PostgreSQL and the app
docker compose up -d --build app          # after changing code or dependencies
docker compose logs -f app
docker compose exec app python scripts/check_ollama.py   # any script runs in the container

# App on the host (development, tests). Stop the container first: both use port 8502.
docker compose up -d milvus postgres
uv sync                                   # install dependencies
uv run alembic upgrade head               # apply DB migrations
uv run streamlit run app/streamlit_app.py # start the UI at http://127.0.0.1:8502
uv run python scripts/documents.py ingest <file>   # add --ocr for scanned files; also: list, delete <id>
uv run python scripts/make_sample_pdf.py  # sample PDF with text, a table and a figure
uv run python scripts/make_sample_pdf.py out.pdf 20   # a 20-page PDF for testing batches
uv run python scripts/search.py "<question>"       # top text and image chunks with scores
uv run python scripts/chat.py "<question>"         # full agent run: trace, thinking, answer
uv run pytest                             # all tests
uv run pytest tests/unit -k chunking      # a subset
uv run ruff check . ; uv run ruff format .
```

Before starting: Tailscale connected on both machines, Ollama running on the Mac with the models
named in `.env` already pulled. Check from Windows: `uv run python scripts/check_ollama.py`

PostgreSQL is published on host port **5433** and Milvus on 19530, both bound to `127.0.0.1`.
Host ports 5432 and 8501 are used by another project on this machine.
Always address them as `127.0.0.1`, never `localhost`: on this machine `localhost` tries IPv6
first and the connection hangs.

New migration after changing `models.py`:
`uv run alembic revision --autogenerate -m "<what changed>"`, review the file, then `upgrade head`.

## Conventions

- **Config**: all tunables (model names, chunk size/overlap, top-k per collection, max retries,
  hosts) live in `Settings` and `.env`. No hard-coded model names or URLs elsewhere.
- **Swappable parts**: to add a chunker, reranker, parser or embedder, implement the interface in
  its package and register it in that package's factory. Callers do not change.
- **Prompts** live in `agent/prompts.py`, not inline in nodes.
- **Types**: type hints everywhere; data crossing a layer boundary is a pydantic model from
  `schemas.py`, not a dict.
- **Tests**: unit tests mock Ollama, Milvus and Postgres. Tests in `tests/integration` need
  `docker compose up`, the migrations applied, and the Mac reachable; they are marked
  `@pytest.mark.integration` (`-m "not integration"` skips them). Postgres tests run inside a
  rolled-back transaction and Milvus tests use throwaway collections, so neither touches real data.
- **Schema changes**: Postgres through an Alembic migration. A Milvus schema change or a change of
  embedding model means dropping the collection and re-ingesting; embedding dimension is read from
  `Settings`.

## Things to know

- Text and image similarity scores are on different scales (CLIP text-to-image cosine is typically
  far lower than text-to-text). Never merge or sort the two result lists by raw vector score; keep
  them as separate top-k lists, in the graph state and in the UI.
- Deleting a document must remove: Milvus rows in both collections, extracted images, the uploaded
  file, and the Postgres `documents` row. Saved retrieval traces keep a snapshot of chunk content,
  so old conversations stay readable.
- Uploading a file whose SHA-256 hash is already registered raises `DuplicateDocumentError`,
  unless the earlier attempt `failed`, in which case the upload replaces it. A failed ingestion
  stays listed with its error.
- Ingestion can be stopped, and how it ends decides what is left:
  - PDFs are parsed `PARSE_PAGE_BATCH` pages at a time (a Docling conversion cannot be
    interrupted). Between page batches, between embedding batches and before storing, the
    pipeline reports an `IngestionProgress` and asks `should_stop`.
  - Deleting a document that is still processing is how it is cancelled: the service's
    `should_stop` is "the registry row is gone". The run ends with `IngestionCancelled` and
    removes what it wrote. Stopping takes up to one batch.
  - A page refresh or closed tab stops the Streamlit script with an exception that is not an
    `Exception` subclass, raised from the progress callback. The pipeline and the service
    therefore clean up on `BaseException`; the document is marked `failed` ("interrupted").
  - At app start `DocumentService.recover_interrupted()` marks any document still
    `processing` as `failed`. It assumes one app process: starting a second one (host and
    container together) would fail the other's running upload.
  - One ingestion at a time per process (`_ingest_lock`); a second upload waits its turn.
- OCR is off unless the upload asks for it (`OCR_DEFAULT`, the checkbox, `--ocr`). There is no
  automatic fallback. With OCR off, text that exists only inside pictures or scanned pages is
  not extracted: a scanned PDF then yields image chunks only (the UI warns), or fails with a
  message pointing at the OCR option if nothing at all is found.
- Tables are stored as Markdown in one chunk. Tables over `TABLE_MAX_CHARS` are split by rows
  with the header row (and caption) repeated.
- The parser merges consecutive text on one page before chunking, so text chunks never span
  pages and each has a page number. Formats without pages (DOCX, MD, TXT) have `page = None`.
- Paths in the database and in Milvus (`documents.file_path`, image chunk `image_path`) are
  relative to the data directory; resolve them with `Settings.resolve_data_path()`.
- Docling and CLIP load lazily on the first ingestion in a process (tens of seconds on CPU);
  the OCR models only on the first upload with OCR on. Listing and deleting documents never
  load them. Parsing itself costs several seconds per page on CPU.
- The first retrieval in a process also loads CLIP (about 28 s measured); later ones take about
  0.3 s. Scripts pay this every run; the Streamlit app should build the retriever once and keep it.
- Vector search always returns its top-k, relevant or not: an off-topic question still gets
  chunks, with low scores (about 0.2 for text on the sample document, against 0.4 to 0.56 for
  on-topic questions). Deciding relevance is the `grade` node's job, not a score cut-off.
- Ollama is remote. `OLLAMA_BASE_URL` in `.env` is the Mac's Tailscale address
  (`http://<mac-tailscale-name>:11434`), never `localhost`. On the Mac, Ollama must be started with
  `OLLAMA_HOST=0.0.0.0`, since it listens only on `127.0.0.1` by default.
- Every LLM and embedding call crosses the network and the Mac may be asleep or loading a model.
  All Ollama clients are built in `llm/` with an explicit timeout and `keep_alive` (an integer
  number of seconds, because `OllamaEmbeddings` rejects strings like `"30m"`); a connection
  failure must surface in the UI as a clear "Ollama unreachable" message, not a stack trace.
- The chat and embedding models share the Mac's 32 GB, and one query uses both.
  If Ollama evicts one to load the other, every query pays a reload; check `ollama ps` on the Mac and
  raise `OLLAMA_MAX_LOADED_MODELS` there if needed.
- `TEXT_EMBED_DIM` (2560) must match what the embedding model returns. The embedder asserts the
  vector length before inserting; changing the model means dropping `text_chunks` and re-ingesting.
- Docker app container:
  - Addresses differ from the host. `docker-compose.yml` sets `MILVUS_URI=http://milvus:19530`,
    `POSTGRES_HOST=postgres`, `POSTGRES_PORT=5432` for the container; these override `.env`,
    which keeps the host values (`127.0.0.1`, 5433). Everything else comes from `.env`.
  - `./data` is mounted at `/app/data`, so host and container share uploads and images. Do not
    ingest from both at the same time.
  - Models are in the `model_cache` volume (`/models`), downloaded on first use.
    `docker compose down -v` deletes it together with the Milvus and PostgreSQL data.
  - PyTorch is CPU-only on Linux through `[tool.uv.sources]` in `pyproject.toml`; `torch` and
    `torchvision` are direct dependencies only so that this applies to them.
  - The container starts with `alembic upgrade head`, so migrations are applied on every start.
  - `.dockerignore` excludes everything and re-includes what the image needs: a new top-level
    folder the app needs must be added there and to the `Dockerfile`.
- Prompts and the CLIP text encoder assume English.
- Model decisions (routing, grading) use `invoke_decision`: the prompt asks for two plain-text
  lines, `Reason: ...` then `Route: ...` / `Relevant: ...`, parsed with a regex. Do not switch
  them to JSON-constrained output (`with_structured_output`): on it `gemma4:e4b-mlx` sometimes
  stops after one field and emits whitespace until the output limit, losing the decision.
  Every model client must have an output limit (`num_predict`): a model that runs away keeps
  streaming, so the request timeout never fires.
- Image sources must be announced in the prompt as attached images (see `source_blocks` and
  `GENERATE_SYSTEM`). Without that, the model answers from the text sources and says the picture
  was "not provided".
- Streamlit specifics that already cost time:
  - `st.status`: do not use it as a `with` block for a live run (leaving the block marks it
    complete), and pass `expanded=True` on every `update()` (an update without it collapses it).
  - `st.chat_input` inside a tab is placed where it is called, not pinned to the bottom. The live
    turn renders into a container created before the input.
  - Delete confirmation is inline, driven by session state, not `st.dialog`: dialog buttons do
    not work under `AppTest`, which is how the UI is tested.
  - Model and document text goes through `safe_markdown()`, which escapes `$` (Streamlit would
    render `$...$` as LaTeX).
  - The UI calls services through `deps.get_chat_service()` / `deps.get_document_service()` at
    call time; tests replace those two functions with fakes.
- Scripts print model output and document text; call `use_utf8_output()` first, because the
  Windows console code page cannot encode all of it.
