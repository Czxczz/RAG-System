"""Composition root.

Builds and caches the singleton orchestrator (and its dependencies) so all
requests share one in-memory FAISS index, embedding model, and registry.
"""
from __future__ import annotations

from functools import lru_cache

from app.config import get_settings, reload_settings
from app.core.auth import reset_auth_service
from app.core.conversation_memory import ConversationStore
from app.core.embeddings import get_embedding_service
from app.core.llm_router import LLMRouter
from app.core.orchestrator import RAGOrchestrator
from app.core.registry import DocumentRegistry
from app.core.reranker import get_reranker
from app.core.vector_store import VectorStore


@lru_cache
def get_conversation_store() -> ConversationStore:
    return ConversationStore()


@lru_cache
def get_orchestrator() -> RAGOrchestrator:
    settings = get_settings()
    embeddings = get_embedding_service()
    store = VectorStore(
        dimension=embeddings.dimension,
        index_path=settings.index_path,
        metadata_path=settings.metadata_path,
    )
    registry = DocumentRegistry(settings.documents_path)
    reranker = get_reranker() if settings.rerank_enabled else None
    return RAGOrchestrator(
        settings=settings,
        embeddings=embeddings,
        store=store,
        registry=registry,
        reranker=reranker,
        conversation_store=get_conversation_store(),
    )


@lru_cache
def get_langchain_rag():
    """LangChain (LCEL) RAG path sharing the orchestrator's components."""
    from app.chains.langchain_rag import build_langchain_rag

    return build_langchain_rag(get_orchestrator())


@lru_cache
def get_langgraph_rag():
    """LangGraph RAG path sharing the orchestrator's components."""
    from app.chains.langgraph_rag import build_langgraph_rag

    return build_langgraph_rag(get_orchestrator())


def apply_runtime_settings() -> None:
    """Reload settings after admin config save and refresh live components."""
    settings = reload_settings()
    reset_auth_service()
    orch = get_orchestrator()
    orch.settings = settings
    orch.llm = LLMRouter(settings)
    orch.rewriter.settings = settings
    orch.rewriter.llm = orch.llm
    orch.query_engine.settings = settings
    # Keep LangChain / LangGraph wrappers on the same settings object.
    try:
        lc = get_langchain_rag()
        lc.settings = settings
        lc.llm = orch.llm
        lc.retriever.top_k = settings.top_k
        lc.retriever.min_score = settings.min_score
    except Exception:  # noqa: BLE001
        pass
    try:
        lg = get_langgraph_rag()
        lg.settings = settings
        lg.llm = orch.llm
        lg._top_k = settings.top_k
        lg._min_score = settings.min_score
    except Exception:  # noqa: BLE001
        pass
