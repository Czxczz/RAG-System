"""Composition root.

Builds and caches the singleton orchestrator (and its dependencies) so all
requests share one in-memory FAISS index, embedding model, and registry.
"""
from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.core.embeddings import get_embedding_service
from app.core.orchestrator import RAGOrchestrator
from app.core.registry import DocumentRegistry
from app.core.vector_store import VectorStore


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
    return RAGOrchestrator(
        settings=settings,
        embeddings=embeddings,
        store=store,
        registry=registry,
    )
