"""Query Engine — turns a natural-language query into ranked context chunks.

    query -> embed -> FAISS search -> (optional rerank) -> threshold filter

When reranking is enabled, FAISS returns a wider candidate pool (`retrieve_k`),
cross-encoder scores reorder them, then the top `top_k` are kept. The vector
similarity floor (`min_score`) is applied before reranking so the threshold
semantics stay cosine-based.
"""
from __future__ import annotations

from app.config import Settings
from app.core.embeddings import EmbeddingService
from app.core.reranker import Reranker
from app.core.vector_store import SearchHit, VectorStore


class QueryEngine:
    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingService,
        store: VectorStore,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.store = store
        self.reranker = reranker

    def retrieve(self, query: str, top_k: int, min_score: float) -> list[SearchHit]:
        pool_k = self.settings.retrieve_k if self.settings.rerank_enabled else top_k
        query_vec = self.embeddings.embed_query(query)
        hits = self.store.search(query_vec, top_k=max(pool_k, top_k))
        hits = [h for h in hits if h.score >= min_score]

        if self.settings.rerank_enabled and self.reranker and hits:
            hits = self.reranker.rerank(query, hits)

        return hits[:top_k]
