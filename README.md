# PrivateRAG AI Assistant

A **private, hybrid RAG-based AI knowledge assistant**. Upload your documents
(PDF / Markdown / TXT) and ask natural-language questions — get **grounded
answers with citations**, powered by a **local or cloud LLM** of your choice.

> Turn any private documents into a trustworthy AI assistant that answers with
> evidence.

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
Client (HTTP)  ─►  FastAPI Gateway  ─►  engine: custom | langchain
                                          │
                        ┌─────────────────┴──────────────────┐
                        ▼                                     ▼
              RAG Orchestrator (default)          LangChainRAG (LCEL)
              app/core/orchestrator.py            app/chains/langchain_rag.py
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
├── chains/              # LangChain (LCEL) wrapper — optional parallel RAG path
│   └── langchain_rag.py # QueryEngineRetriever + LangChainRAG
├── eval/                # Offline eval schemas, metrics, runner
└── core/
    ├── ingestion.py        # load → extract → clean → chunk
    ├── embeddings.py       # local (sentence-transformers) | OpenAI
    ├── vector_store.py     # FAISS index + persisted metadata
    ├── query_rewriter.py   # multi-query expansion (LLM + heuristic + taxonomy)
    ├── query_engine.py     # retrieval pipeline + multi-variant rerank
    ├── reranker.py         # cross-encoder second-stage scoring
    ├── diversity.py        # MMR + hard dedup + text Jaccard dedupe
    ├── context_grouping.py # group chunks by source, trim overlap
    ├── answer_validation.py # post-generation citation validation gate
    ├── llm_router.py       # OpenAI / Gemini / Ollama / extractive fallback
    ├── registry.py         # document catalogue
    └── orchestrator.py     # the core brain

eval/
├── dataset.ec2.json     # Labeled EC2 user guide eval set (28 cases)
├── report.ec2.json      # Latest Gemini eval report (28 cases)
└── report.before.json   # Baseline before redundancy tuning

scripts/
├── run_eval.py          # Offline eval harness (in-process, not HTTP)
├── compare_engines.py   # Side-by-side custom vs LangChain parity check
├── diagnose_retrieval.py # Evidence-based retrieval diagnosis
├── tune_mmr.py          # Sweep MMR lambda / dedup threshold
├── compact_index.py     # Remove exact-duplicate chunks from FAISS index
└── setup_ollama.sh      # Pull recommended Ollama model
```

---

## Quickstart

### 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure (optional)

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

### 3. Run

```bash
uvicorn app.main:app --reload
```

Open the interactive docs at **http://localhost:8000/docs**.

---

## API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Status, providers, document/chunk counts |
| `POST` | `/documents/upload` | Upload & ingest a PDF/MD/TXT file |
| `GET` | `/documents` | List ingested documents |
| `DELETE` | `/documents/{id}` | Delete a document and its chunks |
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
    {"marker": 1, "filename": "notes.pdf", "score": 0.71, "snippet": "..."}
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
- Cloud LLM failures cascade: **OpenAI / Gemini → Ollama → extractive** fallback.

---

## Configuration reference

See [`.env.example`](.env.example). Key settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `EMBEDDING_PROVIDER` | `local` | `local` (private) or `openai` |
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

### Retrieval pipeline

Precision, diversity, and context coherence are improved with a multi-stage
pipeline on top of vector search:

```
query
  → rewrite into variants ........ multi-query (LLM / heuristic / taxonomy)
  → embed + FAISS top RETRIEVE_K .. per variant, fused by max-cosine
  → filter MIN_SCORE
  → cross-encoder rerank .......... max score per chunk across all variants
  → MMR diversity rerank .......... λ·relevance − (1−λ)·redundancy
  → hard cosine dedup ............. drop chunks ≥ MMR_DEDUP_THRESHOLD to a kept chunk
  → text dedupe ................... drop exact / high Jaccard overlap passages
  → context grouping .............. group by source, reading order, trim overlap
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

The project includes a **parallel LangChain (LCEL) path** that wraps the tuned
pipeline without replacing it. This is useful for learning LangChain, adding
streaming/memory later, and migrating toward LangGraph — while keeping one source
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

---

## Roadmap

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
- [ ] LangGraph (retrieve → gate → generate → validate as graph nodes)
- [ ] Web UI (chat + upload)
- [ ] Auth + multi-user isolation
- [ ] Per-document / per-collection scoping

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

**Target redundancy** on a single large PDF: aggregate **0.75–0.82** while
keeping recall@k ≥ 0.8.

Run all cases:

```bash
python scripts/run_eval.py --mode gemini --no-rewrite-llm
python scripts/run_eval.py --mode ollama --no-rewrite-llm
```

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
