# PrivateRAG AI Assistant

A **private, hybrid RAG-based AI knowledge assistant**. Upload your documents
(PDF including scanned/OCR, DOCX, Markdown, TXT) and ask natural-language
questions — get **grounded answers with citations**, powered by a **local or
cloud LLM** of your choice.

> Turn any private documents into a trustworthy AI assistant that answers with
> evidence.

---

## Supported document formats

| Format | Extension | Notes |
| --- | --- | --- |
| PDF (text-based) | `.pdf` | Page numbers preserved in citations |
| PDF (scanned) | `.pdf` | OCR via Tesseract when page text is sparse |
| Word | `.docx` | Paragraphs + tables |
| Plain text | `.txt` | UTF-8 |
| Markdown | `.md`, `.markdown` | UTF-8 |

**Not supported yet:** `.doc` (legacy Word), Excel, PowerPoint.

OCR notes: enabled by default (`OCR_ENABLED=true`). The Docker image includes
Tesseract. For local Python installs, run `brew install tesseract` (macOS) or
`apt install tesseract-ocr` (Debian/Ubuntu).

Upload errors return clear HTTP messages:

| Situation | HTTP |
| --- | --- |
| Unsupported type | `400` |
| Empty file | `400` |
| Over `MAX_UPLOAD_BYTES` (default 25 MB) | `413` |
| Corrupted / unreadable / encrypted / no text | `422` |

---

## Why this exists

| Problem | How PrivateRAG solves it |
| --- | --- |
| Knowledge scattered across files | Unify all documents into one AI interface |
| Keyword search misses meaning | Semantic (vector) search over your content |
| LLMs hallucinate | Grounded-by-default RAG + inline citations |
| Privacy concerns with cloud AI | Hybrid routing: Ollama (local) / OpenAI / Gemini (cloud) |
| Retrieved chunks repeat the same facts | Layered dedup: MMR + hard cosine cap + text overlap filter |

---

## Architecture

```
Client (HTTP)  ─►  FastAPI Gateway  ─►  engine: custom | langchain | langgraph
                                          │
                        ┌─────────────────┴──────────────────┐
                        ▼                                     ▼
              RAG Orchestrator (default)          LangChainRAG / LangGraphRAG
              app/core/orchestrator.py            app/chains/
                        │                                     │
                        └──────────────┬──────────────────────┘
                                       ▼
                              Shared Query Engine + LLM Router
                        (rewrite → FAISS → rerank → MMR → dedup)
                                       │
                        ┌──────────────┴──────────────────────┐
                        ▼                                     ▼
                 Ingestion Pipeline                     Vector DB (FAISS)
              (extract→clean→chunk→embed)
                                       │
                                       ▼
                         Grounded answer + citations + validation
```

Both engines share the same FAISS index, `QueryEngine`, and `LLMRouter`.
The custom path is the default; LangChain is an opt-in parallel composition layer.

### Project layout

```
app/
├── main.py              # FastAPI app + lifespan warm-up
├── config.py            # Env-driven settings (12-factor)
├── models.py            # Pydantic request/response schemas
├── dependencies.py      # Composition root (singletons)
├── api/routes.py        # HTTP endpoints
├── static/              # Web UI (HTML/CSS/JS, served at /)
├── chains/              # LangChain + LangGraph wrappers (optional parallel paths)
│   ├── langchain_rag.py # LCEL: QueryEngineRetriever + LangChainRAG
│   └── langgraph_rag.py # LangGraph: retrieve → gate → generate → validate
├── eval/                # Offline eval schemas, metrics, runner
└── core/
    ├── ingestion.py        # load → extract → OCR (PDF) → clean → chunk
    ├── ocr.py              # Tesseract OCR for scanned PDF pages
    ├── embeddings.py       # local (sentence-transformers) | OpenAI
    ├── vector_store.py     # FAISS index + persisted metadata
    ├── query_rewriter.py   # multi-query expansion (LLM + heuristic + taxonomy)
    ├── query_engine.py     # retrieval pipeline + multi-variant rerank
    ├── reranker.py         # cross-encoder second-stage scoring
    ├── diversity.py        # MMR + hard dedup + text Jaccard dedupe
    ├── context_grouping.py # group chunks by source, trim overlap
    ├── answer_validation.py # post-generation citation validation gate
    ├── prompt_injection.py  # quarantine + jailbreak scan for prompts
    ├── llm_router.py       # OpenAI / Gemini / Ollama / extractive fallback
    ├── registry.py         # document catalogue
    └── orchestrator.py     # the core brain

Dockerfile / docker-compose.yml  # One-command buyer setup

eval/
├── dataset.ec2.json       # Labeled EC2 user guide eval set (28 cases)
├── dataset.multidoc.json  # Multi-doc eval set (17 cases: UG + instance types)
├── report.ec2.json        # Latest Gemini eval report (28 cases)
├── report.multidoc.json   # Latest multi-doc eval report (17 cases)
└── report.before.json     # Baseline before redundancy tuning

scripts/
├── run_eval.py            # Offline eval harness (in-process, not HTTP)
├── ingest_eval_corpus.py  # Ingest multi-doc eval PDFs with canonical names
├── compare_engines.py     # Side-by-side custom vs LangChain parity check
├── diagnose_retrieval.py  # Evidence-based retrieval diagnosis
├── tune_mmr.py            # Sweep MMR lambda / dedup threshold
├── compact_index.py       # Remove exact-duplicate chunks from FAISS index
└── setup_ollama.sh        # Pull recommended Ollama model
```

---

## Requirements & compatibility

| Item | Supported |
| --- | --- |
| **Python** | **3.11.x** (3.11+ recommended; Dockerfile uses `python:3.11-slim`) |
| **OS** | macOS, Linux, Windows (via **Docker Desktop** recommended) |
| **Docker** | Docker Engine / Desktop with Compose v2 |
| **RAM** | ~8 GB minimum for local embeddings + reranker; more if running Ollama locally |
| **Optional LLM** | OpenAI and/or Gemini API keys, or [Ollama](https://ollama.com) on the host |
| **Optional OCR** | Tesseract (bundled in Docker; install locally only for Option B) |

**Windows (local Python):** use PowerShell / cmd:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Install Tesseract from the [UB Mannheim builds](https://github.com/UB-Mannheim/tesseract/wiki) if you need scanned-PDF OCR outside Docker.

**Linux + Ollama in Docker:** `docker-compose.yml` maps `host.docker.internal` via `extra_hosts` so the container can reach Ollama on the host. Ensure Ollama listens on `0.0.0.0:11434` (or set `OLLAMA_BASE_URL` to your host IP).

**Delivery tip (marketplace sales):** ship `.env.example`, never a filled `.env` with real API keys. Buyers copy `.env.example` → `.env`.

---

## Step-by-step setup guide

Follow **Option A** (Docker) unless you specifically need a local Python
dev environment.

### Option A — Docker (recommended for buyers)

1. **Prerequisites:** Docker Desktop (or Docker Engine + Compose v2), ~8 GB RAM.
2. **Configure env:**

   ```bash
   cp .env.example .env
   ```

   Optionally add `OPENAI_API_KEY` and/or `GEMINI_API_KEY`. Without keys,
   extractive answers and local embeddings still work.
3. **Start:**

   ```bash
   docker compose up --build
   ```

4. **Open** [http://localhost:8000/](http://localhost:8000/).
5. **Upload** a PDF/DOCX/TXT/MD, then ask a question in the chat UI.

Uploads and the FAISS index persist in `./data`. Model downloads are cached
in a Docker volume (`model-cache`) so restarts are fast.

> First start downloads the local embedding / reranker models (a few hundred MB)
> and can take several minutes.

**After backend code changes:** rebuild with `docker compose up --build -d`
(static UI files can hot-reload via the bind mount without a rebuild).

### Option B — Local Python

Requires **Python 3.11+**.

#### 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For scanned PDF OCR (optional):

```bash
# macOS
brew install tesseract

# Debian / Ubuntu
sudo apt install tesseract-ocr
```

#### 2. Configure (optional)

```bash
cp .env.example .env
```

Defaults work with **zero configuration**: local embeddings
(`sentence-transformers`) and an extractive fallback if no LLM is configured.

- **Cloud / accuracy mode:** set `OPENAI_API_KEY` (OpenAI) or `GEMINI_API_KEY`
  (Google Gemini) in `.env`.
- **Local / privacy mode:** install [Ollama](https://ollama.com), then:

  ```bash
  chmod +x scripts/setup_ollama.sh
  ./scripts/setup_ollama.sh          # pulls qwen2.5:3b (best for 8 GB RAM)
  ```

  Set `LLM_PROVIDER=ollama` in `.env`, or use `"mode": "ollama"` on `/chat` and eval.

#### 3. Run

```bash
uvicorn app.main:app --reload
```

Open the **Web UI** at **http://localhost:8000/** or API docs at **http://localhost:8000/docs**.

---

## Web UI

A built-in chat interface lives in `app/static/` (no npm build step).

| Feature | Details |
| --- | --- |
| Chat | Multi-turn via `conversation_id` in `localStorage` |
| Streaming | Toggle on → `POST /chat/stream` (token-by-token) |
| Citations | Right panel shows sources (filename + page when available) |
| Upload | Drag-and-drop (multi-file) with progress: extract → embed → index |
| Documents | List with chunk count, size, date; scope checkboxes; admin delete |
| Settings | Per-chat: mode / engine / stream / `top_k` |
| Admin config | API keys + RAG thresholds (saved to `data/runtime_settings.json`) |
| Auth | Optional admin/user login (`AUTH_ENABLED=true`) |

**Note:** streaming always uses the LangChain backend (`/chat/stream`). Turn
streaming off to use `langgraph` or `custom` engines from the UI.

**New conversation** clears the stored `conversation_id` and message history.

---

## API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Status, providers, document/chunk counts |
| `GET` | `/auth/status` | Whether login is required |
| `POST` | `/auth/login` | Obtain session token (admin or user) |
| `GET` | `/auth/me` | Current user + role |
| `GET`/`PUT` | `/admin/config` | Admin-only API keys & RAG thresholds |
| `POST` | `/documents/upload` | Upload & ingest a PDF/DOCX/MD/TXT file |
| `POST` | `/documents/upload/stream` | Same as upload with SSE progress events |
| `GET` | `/documents` | List ingested documents |
| `DELETE` | `/documents/{id}` | Delete a document (admin when auth on) |
| `POST` | `/chat` | Ask a question → grounded answer + citations |

### Example

```bash
# Upload a document
curl -F "file=@notes.pdf" http://localhost:8000/documents/upload

# Ask a question (mode: auto | openai | gemini | ollama | extractive)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings?", "mode": "auto"}'

# Opt-in LangChain (LCEL) path — same retrieval, different composition layer
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings?", "mode": "gemini", "engine": "langchain"}'

# Stream the answer token-by-token via Server-Sent Events (LangChain path)
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings?", "mode": "gemini"}'

# Multi-turn chat — reuse conversation_id from the previous response
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I release it?", "mode": "gemini", "conversation_id": "<id-from-prior-response>"}'
```

| `engine` | Description |
| --- | --- |
| `custom` (default) | Built-in `RAGOrchestrator` — used by eval and default `/chat` |
| `langchain` | LCEL wrapper in `app/chains/` — same `QueryEngine` + `LLMRouter` |
| `langgraph` | LangGraph node graph in `app/chains/langgraph_rag.py` — same components |

`POST /chat/stream` returns `text/event-stream` (SSE) and always uses the
LangChain path. Each `data:` line is a JSON event:

| `type` | Payload | When |
| --- | --- | --- |
| `citations` | `{citations: [...]}` | once, before any tokens (render sources early) |
| `token` | `{text: "..."}` | repeated, incremental answer deltas |
| `done` | `{grounded, provider, validation_notes, conversation_id}` | terminal event |

The retrieval gate refuses *before* streaming (no tokens). The answer
validation gate runs *after* the stream; if it fails, the disclaimer is sent as
a final `token` and `done.grounded` is `false`.

Response:

```json
{
  "answer": "The key findings are ... [1][2]",
  "grounded": true,
  "provider": "gemini",
  "citations": [
    {"marker": 1, "filename": "notes.pdf", "page": 12, "score": 0.71, "snippet": "..."}
  ],
  "validation_notes": [],
  "conversation_id": "a1b2c3..."
}
```

### Chat memory (multi-turn)

Send optional `conversation_id` on `/chat` or `/chat/stream` to continue a
thread. The API returns a `conversation_id` on every response — store it client-side
(e.g. in your Web UI) and send it on the next message.

| Endpoint | Purpose |
| --- | --- |
| `POST /chat` | Ask with memory (custom or langchain engine) |
| `POST /chat/stream` | Stream with memory (LangChain path) |
| `GET /conversations/{id}` | Inspect stored turns |
| `DELETE /conversations/{id}` | Clear a thread |

Memory is **in-process** (like the FAISS index): it survives across requests
while the server is running, but resets on restart. Recent turns are used in two
ways:

1. **Retrieval** — follow-ups like "How do I release it?" are rewritten into a
   standalone search query so FAISS still finds the right chunks.
2. **Generation** — prior user/assistant turns are passed to the LLM so pronouns
   resolve, while the current turn still includes a fresh CONTEXT block with new
   citations.

Configure via `.env`: `CHAT_MEMORY_ENABLED`, `CHAT_MEMORY_MAX_TURNS`,
`CHAT_MEMORY_CONTEXTUALIZE`, `CHAT_MEMORY_CONTEXTUALIZE_USE_LLM`.

---

## Grounding guarantees

- If no chunk clears the similarity threshold (`MIN_SCORE`), or the **retrieval
  confidence gate** rejects low-scoring hits, the API returns **"I couldn't find
  anything relevant…"** and never calls the LLM.
- The system prompt forbids outside knowledge and requires inline `[n]`
  citations mapping to retrieved passages.
- The **answer validation gate** checks that citations exist and are supported;
  failing answers include a disclaimer and `grounded: false`.
- **Prompt injection defense** quarantines user/document text in delimiters,
  strips role spoofing, and (when enabled) blocks common jailbreak phrases
  before the LLM is called.
- Citations include **filename + page** (for PDFs) so answers are traceable.
- Cloud LLM failures cascade: **OpenAI / Gemini → Ollama → extractive** fallback.

---

## Configuration reference

See [`.env.example`](.env.example). Key settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `EMBEDDING_PROVIDER` | `local` | `local` (private) or `openai` |
| `MAX_UPLOAD_BYTES` | `26214400` (25 MB) | Reject larger uploads with HTTP 413 |
| `AUTH_ENABLED` | `false` | Require login; admin vs user roles |
| `AUTH_TOKEN_TTL_SECONDS` | `86400` | Session token lifetime (seconds) |
| `ADMIN_USERNAME` / `ADMIN_PASSWORD` | `admin` / `admin` | Admin account (config + delete) |
| `USER_USERNAME` / `USER_PASSWORD` | `user` / `user` | User account (chat + upload) |
| `OCR_ENABLED` | `true` | OCR sparse/scanned PDF pages via Tesseract |
| `OCR_LANGUAGE` | `eng` | Tesseract language pack |
| `OCR_DPI` | `200` | Rasterization DPI for OCR |
| `OCR_MIN_CHARS_PER_PAGE` | `40` | Native text below this → try OCR |
| `LOCAL_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Hugging Face model for local embeddings |
| `LLM_PROVIDER` | `auto` | `auto` / `openai` / `gemini` / `ollama` / `extractive` |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | — / `gemini-2.5-flash` | Google Gemini cloud |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Local model (recommended for 8 GB RAM) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | Chunking (characters) |
| `TOP_K` | `3` | Chunks sent to the LLM after reranking |
| `MIN_SCORE` | `0.20` | Cosine threshold before reranking |
| `RERANK_ENABLED` | `true` | Enable cross-encoder reranking |
| `RETRIEVE_K` | `20` | FAISS candidate pool when reranking is on |
| `QUERY_REWRITE_ENABLED` | `true` | Multi-query expansion of the question |
| `QUERY_REWRITE_USE_LLM` | `true` | LLM paraphrases (else heuristic fallback) |
| `MMR_ENABLED` | `true` | Diversity-aware (MMR) reranking |
| `MMR_LAMBDA` | `0.6` | Relevance↔diversity trade-off (tuned on EC2 eval) |
| `MMR_DEDUP_THRESHOLD` | `0.85` | Hard cap: drop chunks with cosine ≥ this to a kept chunk |
| `TEXT_DEDUPE_ENABLED` | `true` | Word-overlap text filter after MMR |
| `TEXT_DEDUPE_JACCARD` | `0.85` | Jaccard threshold for near-identical text |
| `CONTEXT_GROUPING_ENABLED` | `true` | Group context by source in reading order |
| `RETRIEVAL_GATE_ENABLED` | `true` | Refuse when best chunk score is below threshold |
| `RETRIEVAL_GATE_MIN_SCORE` | `0.0` | Min post-rerank score (cross-encoder logits) |
| `ANSWER_VALIDATION_ENABLED` | `true` | Validate citations after generation |
| `ANSWER_VALIDATION_MIN_SUPPORT` | `0.5` | Min fraction of citations that must be supported |
| `PROMPT_INJECTION_ENABLED` | `true` | Quarantine prompts + harden system rules |
| `PROMPT_INJECTION_BLOCK` | `true` | Refuse common jailbreak phrases pre-LLM |

### Retrieval pipeline

Precision, diversity, and context coherence are improved with a multi-stage
pipeline on top of vector search:

```
query
  → prompt-injection scan ........ block spoof strip + optional hard block
  → rewrite into variants ........ multi-query (LLM / heuristic / taxonomy)
  → embed + FAISS top RETRIEVE_K .. per variant, fused by max-cosine
  → filter MIN_SCORE
  → cross-encoder rerank .......... max score per chunk across all variants
  → MMR diversity rerank .......... λ·relevance − (1−λ)·redundancy
  → hard cosine dedup ............. drop chunks ≥ MMR_DEDUP_THRESHOLD to a kept chunk
  → text dedupe ................... drop exact / high Jaccard overlap passages
  → context grouping .............. group by source, reading order, trim overlap
  → quarantined prompt ............ UNTRUSTED_CONTEXT / UNTRUSTED_QUESTION tags
  → LLM (grounded answer + [n] citations)
  → answer validation ............. verify citations; disclaimer if unsupported
```

- **Query rewriting** lifts recall by retrieving for several phrasings, then
  fuses the pools (each chunk keeps its best cosine score).
- **MMR + hard dedup** keeps relevant chunks while suppressing near-duplicates
  in embedding space; **text dedupe** catches copy-paste boilerplate MMR misses.
- **Context grouping** presents same-document chunks together in reading order,
  trims overlapping text (sentence + character boundary), and preserves each
  chunk's `[n]` citation marker.

### Index maintenance

Large PDFs (e.g. AWS user guides) can produce repeated headers/footers across
chunks. Ingest skips exact duplicate text; for an existing index:

```bash
python scripts/compact_index.py
```

---

## LangChain integration (optional)

The project includes **parallel LangChain and LangGraph paths** that wrap the
tuned pipeline without replacing it. Useful for streaming, composable chains,
and explicit graph orchestration — while keeping one source
of truth for retrieval quality in `app/core/query_engine.py`.

| Layer | Location | Role |
| --- | --- | --- |
| **Core (tuned)** | `app/core/` | Retrieval, MMR, gates, LLM router — unchanged |
| **LangChain (composition)** | `app/chains/` | `BaseRetriever`, `ChatPromptTemplate`, LCEL `Runnable` |

**What LangChain wraps:** prompt assembly and the generation step as a composable
chain. **What it reuses:** `QueryEngine`, `build_grouped_context`, `LLMRouter`,
and `validate_answer`.

**Streaming** is implemented on this path only (`POST /chat/stream`):
`LangChainRAG.stream()` reuses the retrieval + validation gates and streams LLM
deltas via `LLMRouter.generate_stream()`, which supports OpenAI, Gemini
(`streamGenerateContent`), and Ollama, with the same cloud→Ollama→extractive
fallback as non-streaming generation.

Compare both engines on the same query:

```bash
python scripts/compare_engines.py --mode extractive --limit 5
python scripts/compare_engines.py --case-id imdsv2-require --mode gemini
```

Eval (`scripts/run_eval.py`) still uses the **custom** orchestrator by default.

**LangGraph** (`engine: langgraph`) expresses the same flow as explicit graph
nodes in `app/chains/langgraph_rag.py`: `load_memory` → `retrieve` →
conditional retrieval gate → `build_context` → `generate` → `validate` →
`save_memory`. Compare all three engines with `scripts/compare_engines.py`.

---

## Roadmap

### Shipped (core pipeline)

- [x] Cross-encoder reranking
- [x] Query rewriting (multi-query)
- [x] Diversity-aware reranking (MMR + hard dedup + text dedupe)
- [x] Context grouping with overlap trimming
- [x] Hybrid LLM routing (OpenAI / Gemini / Ollama / extractive)
- [x] Offline eval suite with redundancy metric (28-case EC2 dataset)
- [x] Retrieval confidence gate + answer validation gate
- [x] Taxonomy query heuristic + multi-variant rerank
- [x] LangChain LCEL wrapper (`engine: langchain` on `/chat`)
- [x] Streaming responses (`POST /chat/stream`, SSE, LangChain path)
- [x] Chat memory / multi-turn history (`conversation_id`)
- [x] LangGraph (`engine: langgraph` — retrieve → gate → generate → validate)

### v1 finish line (this repo)

Goal: **end-to-end private document RAG** you can demo locally — chat, upload,
eval on multiple PDFs. No multimodal; that is a separate repo (see below).

| Milestone | Scope |
| --- | --- |
| **Web UI** | ✅ Chat, citation panel, streaming, upload, `conversation_id` (`app/static/`) |
| **Multi-document** | ✅ Per-document scoping on retrieval + upload (`document_ids` on `/chat`) |
| **Multi-doc eval** | ✅ `eval/dataset.multidoc.json` + `scripts/ingest_eval_corpus.py` |
| **LangSmith** (optional) | Dev tracing for LangGraph/UI debugging — not required to ship |
| **Query logging** (optional) | Local SQLite log: query, engine, provider, grounded, latency |

**Remaining before `v1.0`:** merge to `main`, tag **`v1.0`**.

### Next repo — PrivateRAG Multimodal (out of scope for v1)

Not planned in this repository. A future project would cover image / audio /
video ingestion, multimodal embeddings, and unified retrieval — different eval,
compute, and storage requirements than text-only PrivateRAG.

---

## Evaluation

Offline scoring runs the **same in-process pipeline** as `/chat` (no HTTP server
required). The default dataset is the EC2 user guide (`eval/dataset.ec2.json`,
**28 cases**: 23 answerable + 5 refusal).

**Latest report** (`eval/report.ec2.json`, Gemini, `top_k=3`):

| Metric | Score |
| --- | --- |
| precision@k | 0.82 |
| recall@k | 0.81 |
| redundancy | 0.78 |
| citation_accuracy | 1.00 |
| answer_keyword_recall | 0.96 |
| refusal_accuracy | 1.00 |
| hallucination_rate | 0.04 |

Metrics:

| Metric | Meaning |
| --- | --- |
| `precision@k` | Share of top-k retrieved chunks that match `relevant_keywords` |
| `recall@k` | Share of `relevant_keywords` found in top-k chunk text |
| `redundancy` | Max pairwise cosine similarity among final chunks (lower = more diverse) |
| `hallucination_rate` | Answers that fail refusal rules or invent facts/numbers |
| `citation_accuracy` | Share of `[n]` markers that map to a supporting chunk |
| `answer_keyword_recall` | Expected answer phrases present in the response |
| `refusal_accuracy` | Correct "not found" behavior on out-of-corpus questions |
| `source_accuracy` | Retrieved chunks include the expected source PDF (multi-doc eval) |

**Target redundancy** on a single large PDF: aggregate **0.75–0.82** while
keeping recall@k ≥ 0.8.

Run all cases:

```bash
python scripts/run_eval.py --mode gemini --no-rewrite-llm
python scripts/run_eval.py --mode ollama --no-rewrite-llm
```

**Multi-document eval** (User Guide + Instance Types PDFs):

```bash
# Place or ingest both corpora (canonical filenames):
python scripts/ingest_eval_corpus.py --list-expected
python scripts/ingest_eval_corpus.py --file ~/Downloads/ec2-ug.pdf --as ec2-ug.pdf
python scripts/ingest_eval_corpus.py --file ~/Downloads/"Amazon EC2 Instance Types.pdf" --as ec2-types.pdf

python scripts/run_eval.py --dataset eval/dataset.multidoc.json --output eval/report.multidoc.json --mode gemini --no-rewrite-llm
```

**Latest multi-doc report** (`eval/report.multidoc.json`, Gemini, `top_k=3`, 17 cases):

| Metric | Score |
| --- | --- |
| citation_accuracy | 1.00 |
| refusal_accuracy | 1.00 |
| answer_keyword_recall | 1.00 |
| source_accuracy | 1.00 |
| hallucination_rate | 0.12 |
| precision@k | 0.65 |
| recall@k | 0.66 |
| redundancy | 0.80 |

`source_accuracy` measures whether retrieved chunks came from the expected PDF.
Aggregate precision/recall are lower than the single-doc EC2 set because refusal
cases still retrieve chunks (by design).

Run one case at a time and merge into a report (useful on slow hardware or rate-limited APIs):

```bash
python scripts/run_eval.py --list-cases

python scripts/run_eval.py --case-id imdsv2-require --mode gemini --append -v
python scripts/run_eval.py --case-id insufficient-capacity --mode gemini --append -v
# … repeat for each case id
```

Useful flags: `--output eval/report.ec2.json`, `--top-k 3`, `--no-rewrite-llm`,
`--sleep-seconds 15` (Gemini rate limits).

Diagnostic and tuning tools:

```bash
python scripts/diagnose_retrieval.py   # retrieval gap analysis
python scripts/tune_mmr.py           # sweep MMR lambda / dedup threshold
```

Add your own cases by copying the format in `eval/dataset.ec2.json`.

---

## Tests

```bash
pip install pytest
pytest -q
```

---

## FAQ & troubleshooting

### General

**Do I need an API key to try it?**  
No. Local embeddings + extractive answers work with no keys. For higher-quality
answers, add OpenAI and/or Gemini keys, or run Ollama locally.

**Who pays for OpenAI / Gemini / hosting?**  
You do. This package is a **one-time source delivery** with **no ongoing fees**
from the seller. Cloud hosting and LLM API / token usage are billed by those
providers to your accounts. See [License & disclaimer](#license--disclaimer).

**Can I resell or share this source pack?**  
No. The license is **single internal use** for one organization. See `LICENSE`.

### Setup

**First Docker start is slow / stuck downloading models**  
Expected. Embedding and reranker weights download on first boot (hundreds of MB).
Later starts reuse the `model-cache` volume.

**Port 8000 already in use**  
Stop the other process, or change the published port in `docker-compose.yml`
(e.g. `"8001:8000"`) and open that URL instead.

**UI looks stale after a pull**  
Hard-refresh the browser (cache-bust query params are on CSS/JS). After
**Python** backend changes, rebuild: `docker compose up --build -d`.

### LLM providers

**Admin saved an API key but chat still fails / looks unused**  
1. Confirm the key type matches the provider (Gemini keys often start with
   patterns different from OpenAI `sk-…`).  
2. Set **LLM mode** in the sidebar to `gemini` / `openai`, or set
   `LLM_PROVIDER` accordingly (with `auto`, the configured provider is preferred).  
3. Check `/health` for which providers are available.

**Ollama works on the host but not from Docker (Linux)**  
`docker-compose.yml` maps `host.docker.internal` via `extra_hosts`. Ensure
Ollama listens on `0.0.0.0:11434`, or set `OLLAMA_BASE_URL` to your host IP.

**Answers say no LLM is configured**  
No cloud key and Ollama unreachable → extractive fallback. Add a key or start
Ollama, then retry.

### Documents & OCR

**Upload rejected (413 / 422 / 400)**  
- `413`: file over `MAX_UPLOAD_BYTES` (default 25 MB).  
- `400`: unsupported type or empty file.  
- `422`: corrupt, encrypted, or no extractable text.

**Scanned PDF returns little/no text**  
Enable OCR (`OCR_ENABLED=true`). Docker includes Tesseract; local Python needs
`tesseract` installed (`brew` / `apt` / Windows UB Mannheim build).

**Auth login required unexpectedly**  
`AUTH_ENABLED=true` in `.env`. Default accounts are in `.env.example`
(`admin` / `user`). Change passwords before any shared deployment.

---

## License & disclaimer

| Topic | Summary |
| --- | --- |
| **License** | Single-organization **internal use** only. See [`LICENSE`](LICENSE). |
| **No resale** | Do not resell, republish, or redistribute the source (modified or not). |
| **Fees** | One-time purchase of the source pack. **No ongoing license / SaaS fees** from the seller. |
| **Operating costs** | **You** pay cloud hosting, compute, and **LLM / embedding API token costs** (OpenAI, Gemini, etc.). |
| **Copyright** | © 2026 PrivateRAG. All rights reserved. Notices also appear in the Web UI and API (`/api`). |

Full terms: [`LICENSE`](LICENSE).
