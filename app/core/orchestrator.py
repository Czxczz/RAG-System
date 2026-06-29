"""RAG Orchestrator — the core brain.

Coordinates the full lifecycle:

  Ingestion : file -> chunks -> embeddings -> vector store + registry
  Querying  : query -> retrieve -> build grounded prompt -> LLM -> answer + citations

Grounding rules:
  * No relevant context (empty retrieval) => return the canonical
    "Not found in documents" answer; never call the LLM.
  * The system prompt forbids using outside knowledge and requires inline
    [n] citations that map to retrieved chunks.
"""
from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings
from app.core.answer_validation import DISCLAIMER, validate_answer
from app.core.context_grouping import build_flat_context, build_grouped_context
from app.core.embeddings import EmbeddingService
from app.core.ingestion import ingest_file
from app.core.llm_router import LLMRouter
from app.core.query_engine import QueryEngine
from app.core.query_rewriter import QueryRewriter
from app.core.reranker import Reranker
from app.core.registry import DocumentRecord, DocumentRegistry
from app.core.vector_store import SearchHit, StoredChunk, VectorStore

NOT_FOUND_MESSAGE = (
    "I couldn't find anything relevant in your uploaded documents to answer "
    "that question."
)

SYSTEM_PROMPT = """You are PrivateRAG, a careful assistant that answers \
strictly from the provided context.

Rules:
1. Use ONLY the information in the CONTEXT block. Do not use outside knowledge.
2. Every factual statement must cite its source using inline markers like [1] \
or [2], matching the numbered context passages.
3. If the context does not contain the answer, reply exactly: \
"I couldn't find anything relevant in your uploaded documents to answer that \
question."
4. Be concise and precise. Do not invent citations or facts."""


@dataclass
class IngestResult:
    record: DocumentRecord


@dataclass
class AnswerResult:
    answer: str
    grounded: bool
    provider: str
    hits: list[SearchHit]
    # Set by the post-generation validation gate. ``validated`` is True when the
    # answer's citations exist and are sufficiently supported (also True when no
    # validation was performed, e.g. a refusal). ``validation_notes`` explains
    # any problems found.
    validated: bool = True
    validation_notes: list[str] = field(default_factory=list)


class RAGOrchestrator:
    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingService,
        store: VectorStore,
        registry: DocumentRegistry,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.store = store
        self.registry = registry
        self.llm = LLMRouter(settings)
        self.rewriter = QueryRewriter(settings, self.llm)
        self.query_engine = QueryEngine(
            settings,
            embeddings,
            store,
            reranker=reranker if settings.rerank_enabled else None,
            rewriter=self.rewriter if settings.query_rewrite_enabled else None,
        )

    # ── Ingestion ────────────────────────────────────────────
    def ingest(self, source_path: Path, filename: str, content_type: str) -> IngestResult:
        document_id = uuid.uuid4().hex
        chunks = ingest_file(
            source_path,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        if not chunks:
            raise ValueError("No extractable text found in the document.")

        stored = [
            StoredChunk(
                id=f"{document_id}:{c.chunk_index}",
                document_id=document_id,
                filename=filename,
                chunk_index=c.chunk_index,
                page=c.page,
                text=c.text,
            )
            for c in chunks
        ]
        stored = _drop_exact_duplicate_chunks(stored)
        if not stored:
            raise ValueError("No unique text chunks found after de-duplication.")

        vectors = self.embeddings.embed_texts([c.text for c in stored])
        self.store.add(vectors, stored)

        record = DocumentRecord(
            id=document_id,
            filename=filename,
            content_type=content_type,
            num_chunks=len(stored),
            num_chars=sum(len(c.text) for c in stored),
            uploaded_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
        self.registry.add(record)
        return IngestResult(record=record)

    def delete_document(self, document_id: str) -> bool:
        existed = self.registry.remove(document_id)
        self.store.delete_document(document_id)
        return existed

    # ── Querying ─────────────────────────────────────────────
    def answer(self, query: str, mode: str, top_k: int | None = None) -> AnswerResult:
        k = top_k or self.settings.top_k
        hits = self.query_engine.retrieve(
            query, top_k=k, min_score=self.settings.min_score
        )

        # Retrieval confidence gate: refuse deterministically (no LLM call) when
        # nothing was retrieved, or the best chunk is below the score threshold.
        if not hits or self._retrieval_below_gate(hits):
            return AnswerResult(
                answer=NOT_FOUND_MESSAGE, grounded=False, provider="none", hits=[]
            )

        # Context grouping reorders chunks for coherence and returns the
        # marker-aligned ordering, so citations[i] maps to ordered_hits[i].
        if self.settings.context_grouping_enabled:
            context, ordered_hits = build_grouped_context(hits)
        else:
            context, ordered_hits = build_flat_context(hits)

        user_prompt = self._build_prompt(query, context)
        answer, provider = self.llm.generate(SYSTEM_PROMPT, user_prompt, mode=mode)

        return self._validate(answer, provider, ordered_hits)

    def _retrieval_below_gate(self, hits: list[SearchHit]) -> bool:
        if not self.settings.retrieval_gate_enabled:
            return False
        best = max(hit.score for hit in hits)
        return best < self.settings.retrieval_gate_min_score

    def _validate(
        self, answer: str, provider: str, ordered_hits: list[SearchHit]
    ) -> AnswerResult:
        """Run the answer validation gate (if enabled) and assemble the result."""
        if not self.settings.answer_validation_enabled or answer == NOT_FOUND_MESSAGE:
            return AnswerResult(
                answer=answer, grounded=True, provider=provider, hits=ordered_hits
            )

        result = validate_answer(
            answer,
            ordered_hits,
            min_support=self.settings.answer_validation_min_support,
        )
        final_answer = answer if result.passed else answer + DISCLAIMER
        return AnswerResult(
            answer=final_answer,
            grounded=result.passed,
            provider=provider,
            hits=ordered_hits,
            validated=result.passed,
            validation_notes=result.notes,
        )

    @staticmethod
    def _build_prompt(query: str, context: str) -> str:
        return (
            f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\n"
            "Answer with inline [n] citations."
        )


def _drop_exact_duplicate_chunks(chunks: list[StoredChunk]) -> list[StoredChunk]:
    """Skip chunks whose normalized text was already seen in this ingest batch."""
    seen: set[str] = set()
    kept: list[StoredChunk] = []
    for chunk in chunks:
        key = " ".join(chunk.text.lower().split())
        if key in seen:
            continue
        seen.add(key)
        kept.append(chunk)
    return kept
