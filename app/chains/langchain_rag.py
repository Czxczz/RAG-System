"""LangChain (LCEL) re-expression of the PrivateRAG pipeline.

This module wraps the *existing* tuned components as LangChain primitives so the
same retrieval + generation behaviour is available as a composable LCEL chain:

    QueryEngineRetriever  -> BaseRetriever over the tuned QueryEngine
    ChatPromptTemplate    -> the grounded system / user prompt
    RunnableLambda(llm)   -> the hybrid LLM router (OpenAI/Gemini/Ollama/extractive)

The high-level :class:`LangChainRAG` mirrors ``RAGOrchestrator.answer`` — same
retrieval gate, context grouping (marker-aligned citations), and answer
validation gate — but assembled with LCEL. Keeping one source of truth for
retrieval (``QueryEngine``) means the LangChain path and the custom path stay
comparable on the same eval dataset.

Design notes
------------
* We deliberately reuse ``build_grouped_context`` rather than letting LangChain
  format documents, so citation markers ``[n]`` map to the same chunks as the
  custom path.
* The LLM step wraps ``LLMRouter`` (instead of ``ChatOpenAI`` etc.) to preserve
  the cloud→Ollama→extractive fallback cascade that the project relies on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda

from app.config import Settings
from app.core.answer_validation import DISCLAIMER, validate_answer
from app.core.context_grouping import build_flat_context, build_grouped_context
from app.core.llm_router import LLMRouter
from app.core.orchestrator import NOT_FOUND_MESSAGE, SYSTEM_PROMPT
from app.core.vector_store import SearchHit

_HUMAN_PROMPT = (
    "CONTEXT:\n{context}\n\nQUESTION: {query}\n\n"
    "Answer with inline [n] citations."
)


@dataclass
class LCAnswer:
    """Result of a LangChain RAG run (mirrors ``core.orchestrator.AnswerResult``)."""

    answer: str
    grounded: bool
    provider: str
    hits: list[SearchHit]
    validated: bool = True
    validation_notes: list[str] = field(default_factory=list)


def hit_to_document(hit: SearchHit, marker: int) -> Document:
    """Convert a retrieval hit into a LangChain Document with citation metadata."""
    return Document(
        page_content=hit.chunk.text,
        metadata={
            "marker": marker,
            "chunk_id": hit.chunk.id,
            "document_id": hit.chunk.document_id,
            "filename": hit.chunk.filename,
            "page": hit.chunk.page,
            "chunk_index": hit.chunk.chunk_index,
            "score": hit.score,
        },
    )


class QueryEngineRetriever(BaseRetriever):
    """LangChain retriever backed by the project's tuned :class:`QueryEngine`.

    Exposes the full retrieval pipeline (multi-query rewrite, FAISS search,
    cross-encoder rerank, MMR, text dedupe) as a standard ``BaseRetriever`` so it
    can drop into any LangChain chain. Returned Documents carry the same ``[n]``
    citation markers used by the custom path.
    """

    # Typed as Any so any object exposing ``retrieve(query, top_k, min_score)``
    # works (the real QueryEngine, or a stub in tests) — this is a wrapper layer.
    query_engine: Any
    top_k: int
    min_score: float

    # pydantic v2 model config used by BaseRetriever — allow our dataclass types.
    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        hits = self.retrieve_hits(query)
        return [hit_to_document(hit, i) for i, hit in enumerate(hits, start=1)]

    def retrieve_hits(self, query: str) -> list[SearchHit]:
        """Return raw SearchHits (used when callers need scores / grouping)."""
        return self.query_engine.retrieve(
            query, top_k=self.top_k, min_score=self.min_score
        )


class LangChainRAG:
    """High-level RAG runner assembled with LCEL, reusing tuned components.

    Parity with :meth:`RAGOrchestrator.answer`:
      * retrieval confidence gate (deterministic refusal, no LLM call),
      * grouped context with marker-aligned citations,
      * hybrid LLM routing with fallback,
      * post-generation answer validation gate.
    """

    def __init__(
        self,
        settings: Settings,
        query_engine: QueryEngine,
        llm: LLMRouter,
    ) -> None:
        self.settings = settings
        self.query_engine = query_engine
        self.llm = llm
        self.retriever = QueryEngineRetriever(
            query_engine=query_engine,
            top_k=settings.top_k,
            min_score=settings.min_score,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", _HUMAN_PROMPT)]
        )
        self.generation_chain = self._build_generation_chain()

    # ── LCEL assembly ────────────────────────────────────────
    def _build_generation_chain(self) -> Runnable:
        """prompt | llm-router → returns {"answer", "provider"}.

        The router needs the system/user strings, so we render the prompt and
        hand both messages to a RunnableLambda wrapping ``LLMRouter.generate``.
        """

        def call_router(inputs: dict[str, Any]) -> dict[str, str]:
            messages = self.prompt.format_messages(
                context=inputs["context"], query=inputs["query"]
            )
            system_text = messages[0].content
            user_text = messages[1].content
            answer, provider = self.llm.generate(
                system_text, user_text, mode=inputs.get("mode", "auto")
            )
            return {"answer": answer, "provider": provider}

        return RunnableLambda(call_router)

    # ── Public API ───────────────────────────────────────────
    def answer(self, query: str, mode: str = "auto", top_k: int | None = None) -> LCAnswer:
        if top_k is not None and top_k != self.retriever.top_k:
            self.retriever.top_k = top_k

        hits = self.retriever.retrieve_hits(query)
        if not hits or self._retrieval_below_gate(hits):
            return LCAnswer(
                answer=NOT_FOUND_MESSAGE, grounded=False, provider="none", hits=[]
            )

        if self.settings.context_grouping_enabled:
            context, ordered_hits = build_grouped_context(hits)
        else:
            context, ordered_hits = build_flat_context(hits)

        generated = self.generation_chain.invoke(
            {"context": context, "query": query, "mode": mode}
        )
        return self._validate(generated["answer"], generated["provider"], ordered_hits)

    def as_runnable(self) -> Runnable:
        """Expose the end-to-end flow as a single LCEL Runnable.

        Input: ``{"query": str, "mode": str, "top_k": int | None}``
        Output: :class:`LCAnswer`.
        """
        return RunnableLambda(
            lambda x: self.answer(
                x["query"], mode=x.get("mode", "auto"), top_k=x.get("top_k")
            )
        )

    # ── Gates (shared semantics with the custom orchestrator) ─
    def _retrieval_below_gate(self, hits: list[SearchHit]) -> bool:
        if not self.settings.retrieval_gate_enabled:
            return False
        best = max(hit.score for hit in hits)
        return best < self.settings.retrieval_gate_min_score

    def _validate(
        self, answer: str, provider: str, ordered_hits: list[SearchHit]
    ) -> LCAnswer:
        if not self.settings.answer_validation_enabled or answer == NOT_FOUND_MESSAGE:
            return LCAnswer(
                answer=answer, grounded=True, provider=provider, hits=ordered_hits
            )
        result = validate_answer(
            answer,
            ordered_hits,
            min_support=self.settings.answer_validation_min_support,
        )
        final_answer = answer if result.passed else answer + DISCLAIMER
        return LCAnswer(
            answer=final_answer,
            grounded=result.passed,
            provider=provider,
            hits=ordered_hits,
            validated=result.passed,
            validation_notes=result.notes,
        )


def build_langchain_rag(orchestrator: Any) -> LangChainRAG:
    """Build a :class:`LangChainRAG` from an existing orchestrator's components.

    Reuses the orchestrator's query engine and LLM router so both paths share the
    same FAISS index, embeddings, reranker, and provider configuration.
    """
    return LangChainRAG(
        settings=orchestrator.settings,
        query_engine=orchestrator.query_engine,
        llm=orchestrator.llm,
    )
