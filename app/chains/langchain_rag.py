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

from collections.abc import Iterator
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
from app.core.conversation_memory import (
    ConversationStore,
    contextualize_query,
)
from app.core.llm_router import LLMRouter
from app.core.orchestrator import NOT_FOUND_MESSAGE, SYSTEM_PROMPT
from app.core.prompt_injection import (
    BLOCKED_MESSAGE,
    build_grounded_user_prompt,
    prepare_user_query,
)
from app.core.vector_store import SearchHit

_HUMAN_PROMPT = "{user_prompt}"


def _scope_from_ids(document_ids: list[str] | None) -> set[str] | None:
    if not document_ids:
        return None
    return set(document_ids)


@dataclass
class LCAnswer:
    """Result of a LangChain RAG run (mirrors ``core.orchestrator.AnswerResult``)."""

    answer: str
    grounded: bool
    provider: str
    hits: list[SearchHit]
    validated: bool = True
    validation_notes: list[str] = field(default_factory=list)
    conversation_id: str = ""


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


def _citation_dict(marker: int, hit: SearchHit) -> dict[str, Any]:
    """Citation payload for streaming events (mirrors ``models.Citation``)."""
    text = hit.chunk.text
    snippet = text[:280] + ("…" if len(text) > 280 else "")
    return {
        "marker": marker,
        "document_id": hit.chunk.document_id,
        "filename": hit.chunk.filename,
        "page": hit.chunk.page,
        "chunk_id": hit.chunk.id,
        "score": round(hit.score, 4),
        "snippet": snippet,
        "chunk_text": text,
    }


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
    document_ids: set[str] | None = None

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
            query,
            top_k=self.top_k,
            min_score=self.min_score,
            document_ids=self.document_ids,
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
        conversation_store: ConversationStore | None = None,
    ) -> None:
        self.settings = settings
        self.query_engine = query_engine
        self.llm = llm
        self.conversation_store = conversation_store or ConversationStore()
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
                user_prompt=inputs["user_prompt"]
            )
            system_text = messages[0].content
            user_text = messages[1].content
            answer, provider = self.llm.generate(
                system_text, user_text, mode=inputs.get("mode", "auto")
            )
            return {"answer": answer, "provider": provider}

        return RunnableLambda(call_router)

    # ── Public API ───────────────────────────────────────────
    def answer(
        self,
        query: str,
        mode: str = "auto",
        top_k: int | None = None,
        conversation_id: str | None = None,
        document_ids: list[str] | None = None,
    ) -> LCAnswer:
        conv_id, _ = self.conversation_store.get_or_create(conversation_id)
        history = (
            self.conversation_store.recent_turns(
                conv_id, self.settings.chat_memory_max_turns
            )
            if self.settings.chat_memory_enabled
            else []
        )

        if self.settings.prompt_injection_enabled:
            scan = prepare_user_query(
                query, block=self.settings.prompt_injection_block
            )
            query = scan.text
            if self.settings.prompt_injection_block and scan.flagged:
                result = LCAnswer(
                    answer=BLOCKED_MESSAGE,
                    grounded=False,
                    provider="none",
                    hits=[],
                    conversation_id=conv_id,
                    validated=False,
                    validation_notes=[
                        f"Prompt injection blocked: {', '.join(scan.matched)}"
                    ],
                )
                self._remember_exchange(conv_id, query, result.answer)
                return result

        if top_k is not None and top_k != self.retriever.top_k:
            self.retriever.top_k = top_k
        self.retriever.document_ids = _scope_from_ids(document_ids)

        retrieval_query = contextualize_query(query, history, self.settings, self.llm)
        hits = self.retriever.retrieve_hits(retrieval_query)
        if not hits or self._retrieval_below_gate(hits):
            result = LCAnswer(
                answer=NOT_FOUND_MESSAGE,
                grounded=False,
                provider="none",
                hits=[],
                conversation_id=conv_id,
            )
            self._remember_exchange(conv_id, query, result.answer)
            return result

        if self.settings.context_grouping_enabled:
            context, ordered_hits = build_grouped_context(hits)
        else:
            context, ordered_hits = build_flat_context(hits)

        user_prompt = self._build_user_prompt(query, context)
        if history and self.settings.chat_memory_enabled:
            answer, provider = self.llm.generate_with_history(
                SYSTEM_PROMPT, user_prompt, mode=mode, history=history
            )
        else:
            generated = self.generation_chain.invoke(
                {"user_prompt": user_prompt, "mode": mode}
            )
            answer, provider = generated["answer"], generated["provider"]

        result = self._validate(answer, provider, ordered_hits)
        result.conversation_id = conv_id
        self._remember_exchange(conv_id, query, result.answer)
        return result

    def stream(
        self,
        query: str,
        mode: str = "auto",
        top_k: int | None = None,
        conversation_id: str | None = None,
        document_ids: list[str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream the RAG answer as Server-Sent-Event-style dict events.

        Event shapes (``type`` discriminates):
          * ``{"type": "citations", "citations": [...]}`` — emitted once, before
            any tokens, so the client can render sources immediately.
          * ``{"type": "token", "text": str}`` — incremental answer deltas.
          * ``{"type": "done", "grounded": bool, "provider": str,
            "validation_notes": [...]}`` — terminal event.

        The retrieval gate refuses *before* the LLM is called (no token stream).
        The answer validation gate runs *after* the stream completes, since it
        needs the full answer; if it fails, the disclaimer is streamed as a final
        token and ``grounded`` is reported ``False`` in the done event.
        """
        conv_id, _ = self.conversation_store.get_or_create(conversation_id)
        history = (
            self.conversation_store.recent_turns(
                conv_id, self.settings.chat_memory_max_turns
            )
            if self.settings.chat_memory_enabled
            else []
        )

        if top_k is not None and top_k != self.retriever.top_k:
            self.retriever.top_k = top_k
        self.retriever.document_ids = _scope_from_ids(document_ids)

        if self.settings.prompt_injection_enabled:
            scan = prepare_user_query(
                query, block=self.settings.prompt_injection_block
            )
            query = scan.text
            if self.settings.prompt_injection_block and scan.flagged:
                notes = [f"Prompt injection blocked: {', '.join(scan.matched)}"]
                yield {"type": "token", "text": BLOCKED_MESSAGE}
                yield {
                    "type": "done",
                    "grounded": False,
                    "provider": "none",
                    "validation_notes": notes,
                    "conversation_id": conv_id,
                }
                self._remember_exchange(conv_id, query, BLOCKED_MESSAGE)
                return

        retrieval_query = contextualize_query(query, history, self.settings, self.llm)
        hits = self.retriever.retrieve_hits(retrieval_query)
        if not hits or self._retrieval_below_gate(hits):
            yield {"type": "token", "text": NOT_FOUND_MESSAGE}
            yield {
                "type": "done",
                "grounded": False,
                "provider": "none",
                "validation_notes": [],
                "conversation_id": conv_id,
            }
            self._remember_exchange(conv_id, query, NOT_FOUND_MESSAGE)
            return

        if self.settings.context_grouping_enabled:
            context, ordered_hits = build_grouped_context(hits)
        else:
            context, ordered_hits = build_flat_context(hits)

        yield {
            "type": "citations",
            "citations": [
                _citation_dict(i, hit) for i, hit in enumerate(ordered_hits, start=1)
            ],
        }

        user_prompt = self._build_user_prompt(query, context)

        parts: list[str] = []
        provider = "none"
        if history and self.settings.chat_memory_enabled:
            stream_fn = self.llm.generate_stream_with_history(
                SYSTEM_PROMPT, user_prompt, mode, history
            )
        else:
            stream_fn = self.llm.generate_stream(SYSTEM_PROMPT, user_prompt, mode)
        for delta, prov in stream_fn:
            provider = prov
            parts.append(delta)
            yield {"type": "token", "text": delta}
        answer = "".join(parts)

        grounded = True
        notes: list[str] = []
        stored_answer = answer
        if self.settings.answer_validation_enabled and answer != NOT_FOUND_MESSAGE:
            result = validate_answer(
                answer,
                ordered_hits,
                min_support=self.settings.answer_validation_min_support,
            )
            grounded = result.passed
            notes = result.notes
            if not result.passed:
                yield {"type": "token", "text": DISCLAIMER}
                stored_answer = answer + DISCLAIMER

        yield {
            "type": "done",
            "grounded": grounded,
            "provider": provider,
            "validation_notes": notes,
            "conversation_id": conv_id,
        }
        self._remember_exchange(conv_id, query, stored_answer)

    @staticmethod
    def _build_user_prompt(query: str, context: str) -> str:
        return build_grounded_user_prompt(query, context)

    def _remember_exchange(self, conversation_id: str, query: str, answer: str) -> None:
        if not self.settings.chat_memory_enabled:
            return
        self.conversation_store.append_exchange(
            conversation_id,
            query,
            answer,
            max_turns=self.settings.chat_memory_max_turns,
        )

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
        conversation_store=orchestrator.conversation_store,
    )
