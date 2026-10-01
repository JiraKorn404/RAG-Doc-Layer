# PLAN

Build plan for RAG-Doc-Layer. Architecture and conventions are in [CLAUDE.md](CLAUDE.md).

## 1. Goal

A local chatbot that answers questions over uploaded documents containing text, tables and images,
and shows how it got each answer: its steps, the chunks it retrieved, and their scores.

## 2. Decisions

| # | Topic | Decision | Status |
|---|---|---|---|
| D1 | Vector DB | Milvus standalone in Docker | confirmed |
| D2 | Relational DB | PostgreSQL in Docker: conversations, traces, performance, document registry | confirmed |
| D3 | Chat + text embedding | `langchain-ollama` | confirmed |
| D4 | Chunking | Recursive, behind a `Chunker` interface. Phase 5d adds semantic and LLM-based, chosen per upload with tunable parameters; recursive stays the default | confirmed |
| D5 | Retrieval | Vector similarity search only for now; reranking added in Phase 7 | confirmed |
| D6 | Agent | Custom LangGraph `StateGraph` with explicit nodes | confirmed |
| D7 | Frontend | Streamlit, tabs: Chat, Documents | confirmed |
| D8 | Images | Multimodal embeddings in a separate Milvus collection | confirmed |
| D9 | Parser | Docling | confirmed |
| D10 | Reranker | Deferred to Phase 7. Dedicated model covering text and images; candidate `Qwen/Qwen3-VL-Reranker-2B` | deferred, model not chosen |
| D11 | Image embedding model | `sentence-transformers` `clip-ViT-B-32` (512-dim), English-only, runs on the Windows machine | confirmed |
| D12 | Models | main `gemma4:e4b-mlx` (chat, vision, grading); text embedding `qwen3-embedding:4b` (2560-dim) | confirmed, see Q1 |
| D16 | Model host | Ollama on a MacBook M1 Pro 32 GB, reached over Tailscale | confirmed |
| D17 | Language | English for now | confirmed |
| D18 | Users | Single user, no login, conversations not scoped per user | confirmed |
| D13 | UI to backend | Streamlit calls `services/` in-process, no API server | default |
| D14 | Conversation memory | Own `messages` table, last N turns loaded into graph state | default |
| D15 | File types | PDF, DOCX, PPTX, XLSX, CSV, MD, TXT, PNG/JPG | default |

Consequence of D8: Ollama has no image-embedding endpoint, so image vectors come from a non-Ollama
model. This is the one place the project leaves `langchain-ollama`.

Consequence of D5 and D10: with no reranker, text and image hits cannot be put in one ranked list
(their similarity scores are on different scales). Retrieval returns two lists, top `TEXT_TOP_K`
(default 5) and top `IMAGE_TOP_K` (default 3), and both go to the model. The retry decision comes
from a `grade` node (one chat-model call) instead of a rerank-score threshold.

Consequence of D12: `CHAT_MODEL` and `VISION_MODEL` are separate settings that both default to
`gemma4:e4b-mlx`. `TEXT_EMBED_DIM=2560` is the embedding model's native size; the embedder
checks the length it gets back. Qwen3 embeddings
work best with an instruction prefix on the query side only, so the embedder has separate
`embed_query` and `embed_documents` paths.

Consequence of D16: `OLLAMA_BASE_URL` points at the Mac's Tailscale name, the Mac must run Ollama
with `OLLAMA_HOST=0.0.0.0`, and all clients need timeouts and a clear unreachable-state error.

## 3. Data model

### Milvus

`text_chunks` (HNSW, COSINE)

| Field | Type | Notes |
|---|---|---|
| `id` | VARCHAR, PK | uuid |
| `doc_id` | VARCHAR | filter + delete key |
| `chunk_type` | VARCHAR | `text` or `table` |
| `content` | VARCHAR | chunk text, or table as Markdown |
| `page` | INT64 | |
| `chunk_index` | INT64 | order within the document |
| `metadata` | JSON | `filename`, `chunker` (strategy that produced the chunk) |
| `embedding` | FLOAT_VECTOR(`TEXT_EMBED_DIM`) | |

`image_chunks` (HNSW, COSINE)

| Field | Type | Notes |
|---|---|---|
| `id` | VARCHAR, PK | uuid |
| `doc_id` | VARCHAR | |
| `image_path` | VARCHAR | under `data/images/<doc_id>/` |
| `caption` | VARCHAR | Docling caption or nearby text, may be empty |
| `page` | INT64 | |
| `metadata` | JSON | |
| `embedding` | FLOAT_VECTOR(`IMAGE_EMBED_DIM`) | |

### PostgreSQL

| Table | Columns |
|---|---|
| `documents` | `id`, `filename`, `file_hash` (unique), `file_path`, `status` (`processing`/`ready`/`failed`), `error`, `n_text_chunks`, `n_table_chunks`, `n_image_chunks`, `ocr_used`, `chunker`, `chunk_params` (JSONB), `created_at` |
| `conversations` | `id`, `title`, `created_at`, `updated_at` |
| `messages` | `id`, `conversation_id`, `role`, `content`, `reasoning`, `created_at` |
| `retrieved_chunks` | `id`, `message_id`, `chunk_id`, `doc_id`, `chunk_type`, `content_snapshot`, `image_path` (nullable), `metadata` (JSONB), `page`, `similarity_score`, `rerank_score` (nullable, unused until Phase 7), `rank`, `used_in_answer` |
| `trace_events` | `id`, `message_id`, `node`, `summary`, `payload` (JSONB), `started_at`, `duration_ms` |
| `query_metrics` | `id`, `message_id`, `total_ms`, `retrieve_ms`, `grade_ms`, `generate_ms`, `prompt_tokens`, `completion_tokens`, `n_retries`, `chat_model`, `embed_model` |
| `ingestion_metrics` | `id`, `document_id`, `parse_ms`, `chunk_ms`, `embed_ms`, `store_ms`, `total_ms` |

## 4. Pipelines

**Ingestion**: upload -> hash check -> save file -> Docling parse into elements
(`text` / `table` / `image`) -> text through the chunker chosen for the upload, tables as whole Markdown
chunks, images saved to disk -> embed (Ollama for text and tables, CLIP for images) -> insert into
Milvus -> update `documents` and `ingestion_metrics`.

**Query**: `route` -> `rewrite_query` (standalone question from history) -> `retrieve` (top-k from
each collection) -> `grade` (chat model judges whether the context can answer the question) -> if
not and retries remain, back to `rewrite_query`; otherwise `generate` -> `persist`.
When retries run out with no relevant context, `generate` says so instead of guessing.

## 5. Phases

Each phase ends in something runnable. Check items off as they land.

### Phase 0: Scaffold (done 2026-10-01)
- [x] `git init`, `.gitignore`, `pyproject.toml` (uv), ruff + pytest config
- [x] `docker-compose.yml`: Milvus standalone (embedded etcd, local storage), PostgreSQL, named
      volumes, healthchecks. The MinIO images used by the stock Milvus compose file are no longer
      pullable, so Milvus runs as a single container.
- [x] `.env.example` and `config.py` (`Settings`)
- [x] `scripts/check_ollama.py`: reach the Mac over Tailscale, confirm both models are pulled,
      embed one sentence and confirm the vector length is 2560, and send one test
      image to the main model (answers Q1)
- [x] Package skeleton from CLAUDE.md
- **Done when**: `docker compose up -d` is healthy and `uv run pytest` runs an empty suite.

### Phase 1: Storage and models (done 2026-10-01)
- [x] `llm/`: chat, text embedding and image embedding factories
- [x] `storage/milvus_store.py`: create collections, insert, search, delete by `doc_id`
- [x] `storage/postgres/`: models, session, repositories; first Alembic migration
- [x] `schemas.py`: `Chunk`, `RetrievedChunk`, `TraceEvent`
- **Done when**: integration tests round-trip a vector through Milvus and a row through Postgres.

### Phase 2: Ingestion (done 2026-10-01)
- [x] `parser.py`: Docling to `ParsedElement` list, with figure export
- [x] `chunking/`: `Chunker` protocol, recursive implementation, table handling
- [x] `pipeline.py` with timing and cleanup on failure (status updates live in `DocumentService`)
- [x] `DocumentService`: `ingest`, `list_documents`, `delete`
- [x] `scripts/documents.py` (ingest / list / delete) and `scripts/make_sample_pdf.py`
- **Done when**: a script ingests a sample PDF with a table and a figure, and delete leaves nothing behind.

### Phase 3: Retrieval (done 2026-10-01)
- [x] `retriever.py`: dual-collection search returning separate text and image lists of
      `RetrievedChunk` with similarity score (`rerank_score` left `None`)
- [x] `RetrievalResult` schema and `scripts/search.py`
- **Done when**: a script prints the top text and image chunks with similarity scores for a test question.

### Phase 4: Agent (done 2026-10-01)
- [x] `state.py`, `prompts.py`, one file per node (`route`, `direct_answer`, `rewrite_query`,
      `retrieve`, `grade`, `generate`, `persist`), `graph.py`
- [x] Trace events from every node; retry loop with `MAX_RETRIES`
- [x] `ChatService.stream(conversation_id, question)` yielding trace events and answer tokens
- [x] `persist` node writing messages, chunks, trace and metrics
- [x] Citations: the answer cites sources as `[n]`; cited chunks are stored as `used_in_answer`
- [x] `ChatService`: create / list / delete conversations, `get_messages` with sources and trace
- [x] Migration 0002: `retrieved_chunks.metadata` (source filename for stored answers)
- [x] `scripts/chat.py`
- **Done when**: a CLI call streams the trace and a grounded answer, and the rows are in Postgres.

### Phase 5: Streamlit UI (done 2026-10-01)
- [x] Chat tab: history, streaming answer, expandable "Thinking" trace, chunk cards (source, page,
      type, similarity score, rendered table or image; text and image hits in separate groups),
      conversation picker
- [x] Documents tab: uploader with progress, document table (name, status, chunk counts, date),
      delete button per row with confirmation
- [x] Headless app tests (`tests/unit/test_app.py` with fake services,
      `tests/integration/test_app_flow.py` against the real ones)
- **Done when**: the full flow works in the browser from upload to answer to delete.

### Phase 5b: Run the app in Docker (done 2026-10-01)

Goal: `docker compose up -d` starts Milvus, PostgreSQL and the Streamlit app together; the app
is at `http://127.0.0.1:8502`. Ollama stays on the Mac. Running the app on the host with `uv`
keeps working, for development and for the tests.

Checked before planning (2026-10-01): a container on this machine reaches Ollama on the Mac over
Tailscale, both by IP (`100.99.69.11`) and by name (`ds-srs`). The Docker VM has 16 CPUs and
about 16 GB of RAM.

Design:

| Topic | Decision |
|---|---|
| Image | `python:3.12-slim`, dependencies installed with `uv sync --frozen --no-dev` in a layer of its own, then `src/`, `app/`, `migrations/`, `alembic.ini`, `.streamlit/` copied in. Runs as a non-root user. |
| PyTorch | CPU-only wheels on Linux, through a `pytorch-cpu` index in `pyproject.toml` limited to Linux. The current `uv.lock` would install the CUDA build there (several GB, and no GPU in the container). Windows installs are unchanged. |
| System libraries | Only what Docling and OCR need (`libgl1`, `libglib2.0-0`); confirmed by ingesting the sample PDF in the container. |
| Models (CLIP, Docling layout/table, OCR; about 1.1 GB) | Downloaded on first use into the `model_cache` volume, mounted at `/models` (`HF_HOME=/models/hf`; RapidOCR's package model folder is a symlink to `/models/rapidocr`). They survive rebuilds; the first ingestion after creating the volume needs internet. |
| Service addresses | Set in `docker-compose.yml` for the app container: `MILVUS_URI=http://milvus:19530`, `POSTGRES_HOST=postgres`, `POSTGRES_PORT=5432`. Environment variables take precedence over `.env`, so the host values (`127.0.0.1`, port 5433) stay as they are for host runs. |
| Other settings and secrets | `.env` is passed with `env_file` (models, Ollama URL, tunables). It is not copied into the image. |
| Uploaded files and images | `./data` bind-mounted at `/app/data`. Stored paths are already relative to the data directory, so documents ingested on the host stay valid in the container and the other way round. |
| Migrations | `alembic upgrade head` runs before Streamlit starts, as the container command. No shell script file, to avoid Windows line-ending problems. |
| Network exposure | Streamlit listens on `0.0.0.0` inside the container, published as `127.0.0.1:8502` on the host: this machine only. |
| Start order and health | `depends_on` Milvus and PostgreSQL with `condition: service_healthy`; healthcheck on `/_stcore/health`; `restart: unless-stopped`. |
| Build context | `.dockerignore` excludes `.venv`, `.git`, `data`, `tests`, `.env`, caches. |

Steps:
- [x] CPU-only PyTorch index for Linux in `pyproject.toml`; re-locked (no CUDA packages left in
      `uv.lock`); the Windows environment and the test suite are unchanged
- [x] `Dockerfile` (dependency layer, then code layer) and `.dockerignore`
- [x] `app` service and `model_cache` volume in `docker-compose.yml`
- [x] Built: image 3.53 GB; first build about 9.5 minutes, a code-only rebuild well under a minute
- [x] In the container: `scripts/check_ollama.py` passes; the sample PDF was deleted and ingested
      again (3 min 26 s including the one-time model downloads); questions answered through the
      CLI and through the UI on port 8502
- [x] A document ingested on the host is answerable from the container, and one ingested in the
      container is searchable from the host (shared `./data`)
- [x] Container recreated: conversations and documents still there, no model download
- [x] CLAUDE.md and README updated
- **Done when**: met, except that uploading through the containerised UI in a browser was not
  exercised (upload was checked through the CLI in the container and through the headless UI
  test on the host).

Found while verifying, and fixed:
- Routing and grading used JSON-constrained output, on which `gemma4:e4b-mlx` sometimes stops
  after the first field and emits whitespace. They now use a two-line plain-text reply
  (`invoke_decision`).
- The grader is not reliable enough to decide alone that sources are useless (it rejected a
  question about a chart's colours that the answer step answers correctly). The grade now only
  drives the retries; the answer step always receives the last retrieved sources.

Risks:
- Image size: 3.53 GB with CPU PyTorch. Without the CPU index it would be far larger.
- Speed: parsing and CLIP run on CPU inside the Docker VM. Expect ingestion times similar to the
  host, not faster.
- Tailscale: the container depends on Docker Desktop forwarding to the host's Tailscale route.
  It works today; if it stops, the fallback is `OLLAMA_BASE_URL` with the Mac's Tailscale IP.
- Do not run the host app and the container app at the same time against the same documents
  while ingesting: both write `./data` and the same collections.

Decided (2026-10-01): models in a volume, not in the image; the app is published on this
machine only; no live code editing in the container for now (code changes need
`docker compose up -d --build app`).

### Phase 5c: Faster, cancellable ingestion with progress (implemented 2026-10-01)

Why: parsing costs about 6 to 10 s per page on CPU (measured 2026-10-01 on the sample PDF in the
container: 19-21 s with OCR, 12-16 s without, models loaded). A parse cannot be stopped once
started: an upload that was deleted or abandoned kept 5 to 11 cores busy for over 30 minutes.
The UI shows "Parsing document" with no indication of how far it is.

The three parts share one mechanism: **PDFs are parsed in page batches** instead of one call.
Between batches the pipeline can report progress and check whether it should stop.

#### 1. OCR optional

| Item | Design |
|---|---|
| Setting | `OCR_DEFAULT=false` in `Settings` / `.env.example`. |
| Per upload | A checkbox on the Documents tab, "Read text inside images and scanned pages (slower)", initial value from the setting. `DocumentService.ingest(..., ocr: bool | None = None)`; `scripts/documents.py ingest --ocr`. |
| Parser | `DocumentParser.parse(path, image_dir, *, ocr, on_pages=None, should_stop=None)`. One Docling converter per OCR mode, each built on first use. |
| Scanned documents | No automatic fallback: OCR runs only when it is switched on. A PDF or picture that yields no text, table or image with OCR off is marked `failed` with a message that says to upload it again with the OCR checkbox ticked. A scanned PDF with OCR off usually still succeeds, because its pages are indexed as images, but none of its text is searchable; the result summary therefore warns when a document produced no text chunks. |
| Uploaded pictures (PNG/JPG) | Follow the same switch. The picture itself is always indexed as an image. |
| Record | New column `documents.ocr_used` (migration 0003), shown in the documents table. |
| Format coverage | The switch applies to PDFs and uploaded pictures. Other formats (DOCX, PPTX, ...) have no OCR step either way. |

Trade-off to accept: with OCR off, text that exists only inside pictures (axis labels of a chart,
a screenshot of a table) is not extracted as text. The picture is still retrievable as an image.

#### 2. Stop orphaned work

| Item | Design |
|---|---|
| Page batches | PDFs are converted `PARSE_PAGE_BATCH` pages at a time (default 4) with Docling's `page_range`; page count from `pypdfium2` (already installed with Docling). Other formats stay one call. |
| Cancellation check | `IngestionPipeline.run(..., should_stop: Callable[[], bool])`, checked between page batches, between embedding batches and before storing. `DocumentService` passes "the document row no longer exists". Deleting a processing document therefore stops its work at the next batch boundary. |
| Cleanup | On cancel: vectors, extracted images and the upload folder are removed, nothing is marked failed (the row is already gone). New `IngestionCancelled` exception, not reported as an error. |
| Page refresh / closed tab | Streamlit stops the script with an exception that is not an `Exception` subclass, so today the row stays `processing` and nothing is cleaned. The service will clean up in a `finally` and mark the document `failed` with "Upload was interrupted". |
| Restart mid-ingest | On start, any document still `processing` is marked `failed` ("Interrupted by a restart") and its partial data removed: `DocumentService.recover_interrupted()`, called once from `app/deps.py`. This also removes the "delete it before re-uploading" rule, because a failed upload can simply be uploaded again. |
| Time limit | `PARSE_TIMEOUT_S` (default 1800) for a whole document, checked between batches, plus Docling's own `document_timeout` per batch as a backstop for a batch that hangs. |
| One at a time | A process-wide lock in `DocumentService.ingest`: a second upload waits for the first instead of both fighting for the same cores. |

Limit that remains: a batch that has started cannot be interrupted, so stopping takes up to one
batch (roughly 30 to 40 s at the default size).

#### 3. Progress

| Item | Design |
|---|---|
| Callback | `on_progress` receives an `IngestionProgress` (stage, message, current, total, elapsed seconds) instead of a string. |
| Stages reported | "Waiting for another upload to finish", "Parsing pages 5-8 of 37", "Chunking", "Embedding chunks 32 of 120", "Embedding images 3 of 9", "Storing". |
| UI | A progress bar and one status line with elapsed time per file; when done, the per-stage times ("parsed in 48 s, embedded in 9 s") and whether OCR was used. A "Time" column in the documents table, from `ingestion_metrics`. |
| CLI | `scripts/documents.py ingest` prints the same lines. |

#### Steps
- [x] Spike: `page_range` keeps absolute page numbers; a 7-page PDF gave identical elements
      batched and unbatched, with no measurable overhead (33.0 s against 33.8 s)
- [x] `Settings`: `OCR_DEFAULT`, `PARSE_PAGE_BATCH`, `PARSE_TIMEOUT_S`; `.env.example`
- [x] Parser: OCR switch, page batches
- [x] Pipeline: `should_stop`, `IngestionProgress`, `IngestionCancelled`, cleanup on any exit
- [x] Service: `ocr` argument, existence check, interruption handling, `recover_interrupted()`,
      the ingest lock; migration 0003
- [x] UI: OCR checkbox, progress bar and status line, result summary, table columns; CLI flags
- [x] Tests: unit, headless UI, and integration (service behaviour on the real database with a
      fake pipeline; a real 9-page PDF through Docling in batches)
- [x] In the container (CLI): sample PDF parse 19 s with OCR off, 30 s with OCR on (fresh
      process each); a 20-page PDF deleted mid-parse stopped at the next batch boundary, 23 s
      after the delete, leaving nothing in Milvus, PostgreSQL or `./data`; an ingestion killed by
      a container restart was marked failed by `recover_interrupted()` and re-uploaded cleanly
      (20 pages: 108 s parsing, 127 s in total, OCR off)
- [x] CLAUDE.md updated
- [ ] Not done: looking at the progress bar and the OCR checkbox in a real browser, and
      cancelling from a second browser tab (covered by headless UI tests and the CLI only)

#### Found while verifying, not fixed
- On the synthetic 20-page PDF, the container classified five small 3-row tables as pictures
  (so they were indexed as images, not as table text), while the host classified the same pages
  as tables. Same Docling version and settings. The larger table in the sample report is
  detected as a table in both. Cause unknown; worth checking on real documents.

#### Not in this phase
- Background ingestion queue (uploads continue while you chat, survive a page refresh): Phase 6.
- Faster parsing itself (GPU, or parsing on the Mac): would need a different deployment.

#### Decided (2026-10-01)
- OCR is off by default and there is no automatic fallback: OCR runs only when the checkbox (or
  `OCR_DEFAULT`, or `--ocr`) turns it on.
- 4 pages per batch.

### Phase 5d: Chunking strategy chosen per upload (implemented 2026-10-01, verification open)

Goal: on the Documents tab, each upload picks one of three chunking strategies and tunes its
parameters. The choice is recorded with the document and shown in the documents table, so the
effect of a strategy on retrieval can be compared.

Today the chunker is fixed for the whole process: `build_pipeline` calls `get_chunker(settings)`
once and `CHUNKER` / `CHUNK_SIZE` / `CHUNK_OVERLAP` come from `.env`. This phase makes it an
argument of each ingestion, the same way `ocr` already is.

#### 1. What a strategy changes, and what it does not

Only how the **text** elements are split. For all three strategies:

- Tables stay whole Markdown chunks, split by rows over `TABLE_MAX_CHARS`; one chunk per image.
- A strategy works on one text element at a time. The parser already merges text per page, so
  chunks still never span pages and every chunk keeps its page number.
- No chunk exceeds its strategy's maximum size, and none can exceed 16000 characters (Milvus
  `VARCHAR` limit). A piece over the maximum is cut between sentences into parts of similar
  size; only a single sentence that is itself too long is cut with the recursive splitter.
- Sentence splitting (semantic, LLM-based) is a regex and assumes English (D17).

So the table and image handling moves out of `RecursiveChunker` into a shared `BaseChunker`,
and each strategy implements only `split_texts(texts, checkpoint) -> list[list[str]]`: all text
elements of the document at once, because the semantic threshold is document-wide.

#### 2. The strategies and their parameters

Defaults come from `Settings` / `.env`; the UI starts from them and each upload can override.

**Recursive** (current behaviour, stays the default). No model calls.

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `chunk_size` | 1000 | 200 - 16000 | Maximum characters per chunk |
| `chunk_overlap` | 150 | 0 - below `chunk_size` | Characters repeated between neighbouring chunks |

**Semantic**. Splits where the topic changes: sentences are embedded with the text embedding
model (document path, no query instruction), and a chunk boundary is placed where the cosine
distance between neighbouring sentence groups is unusually large for that document.

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `breakpoint_percentile` | 90 | 50 - 99 | A boundary is placed where the distance is above this percentile of all distances in the document. Lower gives more, smaller chunks |
| `buffer_sentences` | 1 | 0 - 3 | Sentences on each side joined to a sentence before embedding it, to smooth out very short sentences |
| `min_chunk_chars` | 200 | 0 - 2000 | A smaller chunk is merged into its neighbour |
| `max_chunk_chars` | 2000 | 500 - 16000 | A larger chunk is cut between sentences |

Written in this project (about a hundred lines) rather than taken from `langchain_experimental`:
it needs batching, progress and cancellation between embedding batches, and that package is a
new dependency for one class.

**LLM-based**. The chat model decides the boundaries. The text is cut into numbered sentences;
the model sees a window of them and replies with the numbers where a new topic starts. The
chunks are then assembled from the original text.

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `target_chunk_chars` | 1000 | 300 - 8000 | Size the prompt asks the model to aim for. A page no longer than this is one chunk, without a model call |
| `max_chunk_chars` | 2000 | 500 - 16000 | A larger chunk is cut between sentences |
| `window_chars` | 6000 | 2000 - 16000 | Text shown to the model per call. Larger means fewer calls but a longer prompt |

Design points that follow from what is already known about `gemma4:e4b-mlx`:

- The model returns **boundaries, never text**. It cannot alter or invent document content, and
  the reply stays a few tokens long.
- Reply format is two plain-text lines (`Reason: ...` / `Splits: 4, 9, 15`) parsed with a regex,
  like `invoke_decision`. No JSON-constrained output.
- A new `chunking_model` client in `llm/`: no thinking, temperature 0, output limit
  `CHUNKING_MAX_TOKENS` (default 200). The prompt goes in a new
  `ingestion/chunking/prompts.py`, not `agent/prompts.py`: `ingestion/` does not import from
  `agent/`.
- Numbers outside the window or not increasing are dropped. A reply with no usable line means
  that window is split by size instead (between sentences, at the target size); the number of
  such windows is counted and reported in the upload summary, so a strategy that silently
  degraded is visible.
- A sentence is shown to the model on one line, cut at 500 characters, which bounds the prompt
  for text without sentence punctuation.
- When a window ends mid-topic, the sentences after its last boundary start the next window.
- Ollama unreachable fails the ingestion with the usual message (same as for embedding).

#### 3. Cost

| Strategy | Extra work per document | Expectation |
|---|---|---|
| Recursive | none | milliseconds |
| Semantic | one embedding per sentence, in batches of `EMBED_BATCH_SIZE`, on top of embedding the final chunks | measured: about 0.3 s per sentence (5 s per batch of 16); 169 sentences on 8 pages of prose took 53 s |
| LLM-based | about one chat call per page (a page usually fits one window) | measured: 1 to 4 s per call; 8 pages took 33 s in one run, 4 pages 4.5 s in another |

Both are comparable to or smaller than parsing (6 to 10 s per page). Both alternate with or
reuse the models on the Mac; LLM-based chunking loads the chat model during ingestion, then the
embedding model, so watch `ollama ps` for eviction.

#### 4. Design

| Item | Design |
|---|---|
| Config model | `ChunkingConfig` in `schemas.py`: a discriminated union on `strategy` (`recursive` / `semantic` / `llm`) of three pydantic models holding the parameters above, with the ranges as validators. This is what crosses every layer, and what is stored. |
| Settings | `CHUNKER` stays the default strategy. New defaults: `SEMANTIC_BREAKPOINT_PERCENTILE`, `SEMANTIC_BUFFER_SENTENCES`, `SEMANTIC_MIN_CHUNK_CHARS`, `SEMANTIC_MAX_CHUNK_CHARS`, `LLM_CHUNK_TARGET_CHARS`, `LLM_CHUNK_MAX_CHARS`, `LLM_CHUNK_WINDOW_CHARS`, `CHUNKING_MAX_TOKENS`. `Settings.chunking_defaults()` builds one `ChunkingConfig` per strategy from them and `default_chunking()` picks the `CHUNKER` one; an out-of-range default fails when the settings load. `TABLE_MAX_CHARS` stays a setting only, not in the UI. |
| Chunker interface | `Chunker.chunk(elements, *, doc_id, metadata, checkpoint=None)`. `checkpoint(message, current, total)` is the pipeline's existing stop-and-report hook, called between embedding batches (semantic) and between model calls (LLM), so progress shows "Chunking: embedding sentences (32 of 169 done)" and a deleted upload stops within one batch. |
| Factory | `get_chunker(config, settings)` in `chunking/factory.py`; `CHUNKERS` maps strategy name to a builder taking the config. The semantic builder gets the text embedder, the LLM builder the chunking model. Adding a fourth strategy is still: implement, add a params model, register. |
| Pipeline | `IngestionPipeline.run(..., chunking: ChunkingConfig | None = None)`; `None` means the configured default. The chunker is built per run instead of once in `build_pipeline` (construction is cheap; the embedder and model clients are reused). `IngestionResult` gains `chunking` and `n_chunk_fallbacks`. |
| Service | `DocumentService.ingest(..., chunking: ChunkingConfig | None = None)`, plus `chunking_defaults` (one default config per strategy) and `default_chunker` for the UI. `DocumentInfo` gains `chunker`, `chunk_params` and, on the value `ingest` returns, `chunk_fallbacks`. |
| Record | Migration 0004: `documents.chunker` (VARCHAR, existing rows `recursive`) and `documents.chunk_params` (JSONB, nullable), written when the upload is registered, so a failed upload shows them too. Each chunk's Milvus `metadata` JSON also gets `chunker`, so text chunk cards in the Chat tab show which strategy produced a retrieved chunk. No Milvus schema change, no re-ingestion. |
| UI | Under the uploader: a select box "Chunking strategy" (Recursive / Semantic / LLM-based) with a one-line description and cost note, then an expander "Parameters" showing that strategy's number inputs with a help text each. The inputs are generated from the strategy's model (field title, description, bounds), so a new strategy needs no UI code. Not inside `st.form`, because the parameter widgets must change when the strategy changes. Widget keys include the strategy, so each keeps its own values. The settings apply to all files of one upload. Invalid combinations (overlap not below size, min above max) disable the upload button with a message. |
| Documents table | New "Chunking" column (strategy name; parameters in its tooltip). The upload summary names the strategy, and warns if LLM windows fell back to recursive. |
| CLI | `scripts/documents.py ingest <file> --chunker semantic --chunk-param breakpoint_percentile=85` (repeatable). |
| Layering | `app/` builds a `ChunkingConfig` from `ragdoc.schemas` and passes it to the service; it imports nothing else. |

#### 5. Steps

- [x] Spike on 8 pages of a real document (page text read with pypdfium2): 7 to 44 sentences
      per page, median sentence 137 characters. Semantic: 53 s, 25 chunks, median 842
      characters. LLM-based: 8 calls, 33 s, 36 chunks, median 548, boundaries at the section
      headings, no unreadable reply. Defaults kept. The sample PDFs are too small to show a
      difference: their pages are under 1000 characters, so each page is one chunk
- [x] `ChunkingConfig` and the params models in `schemas.py`; new `Settings` fields,
      `chunking_defaults()` and `default_chunking()`; `.env.example`
- [x] Shared base for tables and images; `RecursiveChunker` on top of it with unchanged output
- [x] Sentence splitter; `SemanticChunker`
- [x] `get_chunking_model` in `llm/chat.py`; prompt; `LLMChunker` with reply parsing and fallback
- [x] Factory taking a config; `checkpoint` in the `Chunker` protocol
- [x] Pipeline and service arguments; migration 0004 (applied); repository and `DocumentInfo`
- [x] UI: strategy select, parameter expander, table column, summary, chunk cards; CLI flags
- [x] Unit and headless UI tests: 116 pass (`-m "not integration"`)
- [ ] Integration tests: written (`test_text_is_chunked_by_the_chosen_strategy` with the real
      models, service tests for the recorded choice) but **not yet run to completion**. One
      run hit the 10-minute limit, the next was stopped on request
- [x] App container rebuilt with this phase; it starts, applies nothing new (0004 already
      applied from the host) and lists the existing documents as `recursive`
- [ ] In the container: one upload per strategy through the UI, then the same question against
      each with `scripts/search.py`. Not done
- [ ] Cancelling an upload during semantic or LLM chunking on a real document (unit-tested only)
- [x] CLAUDE.md and D4 updated
- **Done when**: the same file can be indexed with each of the three strategies from the
  browser, the documents table shows which one was used with which parameters, and deleting an
  upload during semantic or LLM chunking stops it.

Changed from the plan while building:
- A chunk over the maximum is cut between sentences, not with the recursive splitter: the spike
  showed chunks ending in the middle of a sentence.
- Parameters are number inputs generated from the strategy's model, not hand-written widgets
  with a slider.
- The semantic threshold is one percentile over the whole document, so a strategy receives all
  text elements at once (`split_texts`).

#### Not in this phase

- **Re-chunking an indexed document.** Uploading a file again with another strategy still raises
  `DuplicateDocumentError`; to try another strategy, delete the document and upload it again,
  which parses it again. A "re-index with other settings" action that reuses the stored upload
  (and ideally cached parsed elements) is the natural follow-up, see Q4.
- Chunks spanning pages. A topic that continues over a page break is still cut there.
- Rewriting or summarising chunks with the model (contextual headers, propositions).
- A strategy for tables or images.

### Phase 6: Hardening
- [ ] Performance view (latency per node, tokens) read from `query_metrics`
- [ ] Error states in the UI (Ollama down, model not pulled, Milvus unreachable)
- [ ] A small evaluation set with expected source documents, and a script reporting hit rate
- [ ] README with setup steps

### Phase 7: Reranking (not started, build after Phases 0-6)
- [ ] Choose the model. Candidates: `Qwen/Qwen3-VL-Reranker-2B` (one model for text, tables and
      images, Apache 2.0), `jinaai/jina-reranker-m0` (same idea, non-commercial licence), or
      `BAAI/bge-reranker-v2-m3` for text plus one of the above for images
- [ ] Decide where it runs. Ollama has no rerank endpoint, so either in the app process (needs a
      GPU on the Windows machine) or as a small rerank service on the Mac over Tailscale
- [ ] `retrieval/reranker.py`: `Reranker` interface, implementation, factory
- [ ] `rerank` node between `retrieve` and `grade`; retrieve more candidates, keep the top few
- [ ] Fill `rerank_score`; show it in the chunk cards; merge text and image hits into one ranked list
- [ ] Add `rerank_ms` to `query_metrics` (migration)
- [ ] Compare hit rate with and without reranking on the Phase 6 evaluation set
- **Done when**: the UI shows both scores and the evaluation script reports the difference.

### Later
- Other chunkers (layout-aware) via the `Chunker` interface; semantic and LLM-based are Phase 5d
- Hybrid search (BM25 + dense) in Milvus
- Image captions as extra text chunks, to complement image embeddings
- FastAPI layer over `services/`; background ingestion queue

## 6. Open questions

- ~~**Q1. Does `gemma4:e4b-mlx` accept images through Ollama?**~~ Resolved 2026-10-01: yes.
  `scripts/check_ollama.py` sent it a test image and it answered correctly.
- **Q4. Comparing chunking strategies on one document (Phase 5d).** As planned, trying another
  strategy means delete and upload again, paying the parse again. If comparing is the main use,
  a re-index action that reuses the upload should move into the phase.
- **Q2. Scale.** Expected number and size of documents. Large uploads would move ingestion to a
  background worker sooner.
- **Q3. Other languages later.** `qwen3-embedding` is multilingual, but `clip-ViT-B-32` is not.
  Moving beyond English means swapping the image embedder (for example `jinaai/jina-clip-v2`) and
  re-ingesting images.
