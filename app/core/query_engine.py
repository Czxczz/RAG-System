"""Query Engine — turns a natural-language query into ranked context chunks.

    query -> embed -> FAISS search -> threshold filter -> ranked hits

Reranking is intentionally left as a v2 hook (see `rerank`).
"""
from __future__ import annotations

from app.core.embeddings import EmbeddingService
from app.core.vector_store import SearchHit, VectorStore


class QueryEngine:
    def __init__(self, embeddings: EmbeddingService, store: VectorStore) -> None:
        self.embeddings = embeddings
        self.store = store

    def retrieve(self, query: str, top_k: int, min_score: float) -> list[SearchHit]:
        query_vec = self.embeddings.embed_query(query)
        hits = self.store.search(query_vec, top_k=top_k)
        hits = self.rerank(query, hits)
        return [h for h in hits if h.score >= min_score]

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        """v2 hook for a cross-encoder reranker. Identity for the MVP."""
        return hits
