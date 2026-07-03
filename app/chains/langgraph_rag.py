"""LangGraph re-expression of the PrivateRAG pipeline.

Composes the same tuned components as :class:`LangChainRAG` as explicit graph
nodes with conditional routing:

    load_memory → retrieve → {refuse | build_context → generate → validate}
    → save_memory

Each node reuses existing functions (``QueryEngine``, ``LLMRouter``, gates,
``ConversationStore``) so behaviour stays comparable across custom, LangChain,
and LangGraph paths.
"""
from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.chains.langchain_rag import LCAnswer
from app.config import Settings
from app.core.answer_validation import DISCLAIMER, validate_answer
from app.core.context_grouping import build_flat_context, build_grouped_context
from app.core.conversation_memory import (
    ChatTurn,
    ConversationStore,
    contextualize_query,
)
from app.core.llm_router import LLMRouter
from app.core.orchestrator import NOT_FOUND_MESSAGE, SYSTEM_PROMPT
from app.core.vector_store import SearchHit

RouteAfterRetrieve = Literal["refuse", "continue"]


class GraphState(TypedDict, total=False):
    """Mutable state passed between graph nodes."""

    query: str
    mode: str
    top_k: int | None
    conversation_id: str | None
    document_ids: list[str] | None
    conv_id: str
    history: list[ChatTurn]
    retrieval_query: str
    hits: list[SearchHit]
    ordered_hits: list[SearchHit]
    context: str
    answer: str
    provider: str
    grounded: bool
    validated: bool
    validation_notes: list[str]
    refused: bool


class LangGraphRAG:
    """RAG runner with retrieve → gate → generate → validate as graph nodes."""

    def __init__(
        self,
        settings: Settings,
        query_engine: Any,
        llm: LLMRouter,
        conversation_store: ConversationStore | None = None,
    ) -> None:
        self.settings = settings
        self.query_engine = query_engine
        self.llm = llm
        self.conversation_store = conversation_store or ConversationStore()
        self._top_k = settings.top_k
        self._min_score = settings.min_score
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(GraphState)
        builder.add_node("load_memory", self._load_memory)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("refuse", self._refuse)
        builder.add_node("build_context", self._build_context)
        builder.add_node("generate", self._generate)
        builder.add_node("validate", self._validate)
        builder.add_node("save_memory", self._save_memory)

        builder.add_edge(START, "load_memory")
        builder.add_edge("load_memory", "retrieve")
        builder.add_conditional_edges(
            "retrieve", self._route_after_retrieve, {"refuse": "refuse", "continue": "build_context"}
        )
        builder.add_edge("refuse", "save_memory")
        builder.add_edge("build_context", "generate")
        builder.add_edge("generate", "validate")
        builder.add_edge("validate", "save_memory")
        builder.add_edge("save_memory", END)
        return builder.compile()

    # ── Public API ───────────────────────────────────────────
    def answer(
        self,
        query: str,
        mode: str = "auto",
        top_k: int | None = None,
        conversation_id: str | None = None,
        document_ids: list[str] | None = None,
    ) -> LCAnswer:
        state = self.graph.invoke(
            {
                "query": query,
                "mode": mode,
                "top_k": top_k,
                "conversation_id": conversation_id,
                "document_ids": document_ids,
            }
        )
        return LCAnswer(
            answer=state["answer"],
            grounded=state["grounded"],
            provider=state["provider"],
            hits=state.get("ordered_hits") or state.get("hits") or [],
            validated=state.get("validated", True),
            validation_notes=state.get("validation_notes") or [],
            conversation_id=state["conv_id"],
        )

    # ── Graph nodes ──────────────────────────────────────────
    def _load_memory(self, state: GraphState) -> GraphState:
        conv_id, _ = self.conversation_store.get_or_create(state.get("conversation_id"))
        history: list[ChatTurn] = []
        if self.settings.chat_memory_enabled:
            history = self.conversation_store.recent_turns(
                conv_id, self.settings.chat_memory_max_turns
            )
        return {"conv_id": conv_id, "history": history}

    def _retrieve(self, state: GraphState) -> GraphState:
        history = state.get("history") or []
        query = state["query"]
        retrieval_query = contextualize_query(query, history, self.settings, self.llm)
        top_k = state.get("top_k") if state.get("top_k") is not None else self._top_k
        scope = _scope_from_ids(state.get("document_ids"))
        hits = self.query_engine.retrieve(
            retrieval_query,
            top_k=top_k,
            min_score=self._min_score,
            document_ids=scope,
        )
        refused = not hits or self._retrieval_below_gate(hits)
        return {
            "retrieval_query": retrieval_query,
            "hits": hits,
            "ordered_hits": [],
            "refused": refused,
        }

    @staticmethod
    def _route_after_retrieve(state: GraphState) -> RouteAfterRetrieve:
        return "refuse" if state.get("refused") else "continue"

    def _refuse(self, state: GraphState) -> GraphState:
        return {
            "answer": NOT_FOUND_MESSAGE,
            "provider": "none",
            "grounded": False,
            "validated": True,
            "validation_notes": [],
            "ordered_hits": [],
        }

    def _build_context(self, state: GraphState) -> GraphState:
        hits = state.get("hits") or []
        if self.settings.context_grouping_enabled:
            context, ordered_hits = build_grouped_context(hits)
        else:
            context, ordered_hits = build_flat_context(hits)
        return {"context": context, "ordered_hits": ordered_hits}

    def _generate(self, state: GraphState) -> GraphState:
        user_prompt = _build_user_prompt(state["query"], state["context"])
        history = state.get("history") or []
        mode = state.get("mode", "auto")
        if history and self.settings.chat_memory_enabled:
            answer, provider = self.llm.generate_with_history(
                SYSTEM_PROMPT, user_prompt, mode=mode, history=history
            )
        else:
            answer, provider = self.llm.generate(SYSTEM_PROMPT, user_prompt, mode=mode)
        return {"answer": answer, "provider": provider}

    def _validate(self, state: GraphState) -> GraphState:
        answer = state["answer"]
        provider = state["provider"]
        ordered_hits = state.get("ordered_hits") or []
        if not self.settings.answer_validation_enabled or answer == NOT_FOUND_MESSAGE:
            return {
                "answer": answer,
                "grounded": True,
                "validated": True,
                "validation_notes": [],
            }
        result = validate_answer(
            answer,
            ordered_hits,
            min_support=self.settings.answer_validation_min_support,
        )
        final_answer = answer if result.passed else answer + DISCLAIMER
        return {
            "answer": final_answer,
            "grounded": result.passed,
            "validated": result.passed,
            "validation_notes": result.notes,
        }

    def _save_memory(self, state: GraphState) -> GraphState:
        if self.settings.chat_memory_enabled:
            self.conversation_store.append_exchange(
                state["conv_id"],
                state["query"],
                state["answer"],
                max_turns=self.settings.chat_memory_max_turns,
            )
        return {}

    def _retrieval_below_gate(self, hits: list[SearchHit]) -> bool:
        if not self.settings.retrieval_gate_enabled:
            return False
        best = max(hit.score for hit in hits)
        return best < self.settings.retrieval_gate_min_score


def _build_user_prompt(query: str, context: str) -> str:
    return (
        f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\n"
        "Answer with inline [n] citations."
    )


def _scope_from_ids(document_ids: list[str] | None) -> set[str] | None:
    if not document_ids:
        return None
    return set(document_ids)


def build_langgraph_rag(orchestrator: Any) -> LangGraphRAG:
    """Build :class:`LangGraphRAG` from an existing orchestrator's components."""
    return LangGraphRAG(
        settings=orchestrator.settings,
        query_engine=orchestrator.query_engine,
        llm=orchestrator.llm,
        conversation_store=orchestrator.conversation_store,
    )
