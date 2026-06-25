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
| Privacy concerns with cloud AI | Hybrid local (Ollama) / cloud (OpenAI) routing |

---

## Architecture

```
Frontend (v2)  ─►  FastAPI Gateway  ─►  RAG Orchestrator (core brain)
                                          │
                        ┌─────────────────┴──────────────────┐
                        ▼                                     ▼
                 Ingestion Pipeline                     Query Engine
              (extract→clean→chunk→embed)         (embed→FAISS search→rank)
                        │                                     │
                        ▼                                     ▼
                 Embedding Service  ◄────────────────►  Vector DB (FAISS)
                                                              │
                                                              ▼
                                                       LLM Router
                                                   (local / cloud / auto)
                                                              │
                                                              ▼
                                                Grounded answer + citations
```

Every component is modular and independently replaceable.

### Project layout

```
app/
├── main.py              # FastAPI app + lifespan warm-up
├── config.py            # Env-driven settings (12-factor)
├── models.py            # Pydantic request/response schemas
├── dependencies.py      # Composition root (singletons)
├── api/routes.py        # HTTP endpoints
└── core/
    ├── ingestion.py     # load → extract → clean → chunk
    ├── embeddings.py    # local (sentence-transformers) | OpenAI
    ├── vector_store.py  # FAISS index + persisted metadata
    ├── query_engine.py  # retrieval (+ v2 rerank hook)
    ├── llm_router.py    # OpenAI / Gemini / Ollama / extractive fallback
    ├── registry.py      # document catalogue
    └── orchestrator.py  # the core brain
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
- **Local / privacy mode:** run [Ollama](https://ollama.com) (`ollama pull llama3.1`).

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
```

Response:

```json
{
  "answer": "The key findings are ... [1][2]",
  "grounded": true,
  "provider": "openai",
  "citations": [
    {"marker": 1, "filename": "notes.pdf", "score": 0.71, "snippet": "..."}
  ]
}
```

---

## Grounding guarantees

- If no chunk clears the similarity threshold (`MIN_SCORE`), the API returns
  **"I couldn't find anything relevant…"** and never calls the LLM.
- The system prompt forbids outside knowledge and requires inline `[n]`
  citations mapping to retrieved passages.

---

## Configuration reference

See [`.env.example`](.env.example). Key settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `EMBEDDING_PROVIDER` | `local` | `local` (private) or `openai` |
| `LLM_PROVIDER` | `auto` | `auto` / `openai` / `gemini` / `ollama` / `extractive` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | Chunking (characters) |
| `TOP_K` | `5` | Chunks retrieved per query |
| `MIN_SCORE` | `0.20` | Cosine threshold for "no context" |

---

## Roadmap (v2+)

- [ ] Cross-encoder reranking (`query_engine.rerank` hook is ready)
- [ ] Web UI (chat + upload)
- [ ] Auth + multi-user isolation
- [ ] Streaming responses
- [ ] Per-document / per-collection scoping

---

## Tests

```bash
pip install pytest
pytest -q
```
