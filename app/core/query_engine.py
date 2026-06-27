"""Query Engine — turns a natural-language query into ranked context chunks.

Pipeline:

    query
      -> rewrite into variants (multi-query)            [query_rewriter]
      -> embed variants + FAISS search, fuse by max-cosine
      -> cosine threshold filter (min_score)
      -> cross-encoder rerank                            [reranker]
      -> diversity-aware reranking (MMR)                 [diversity]
      -> top_k

Each stage is independently toggle-able via settings, so the engine degrades
gracefully (e.g. no LLM -> heuristic query rewriting; rerank off -> cosine
order feeds MMR directly).
"""
from __future__ import annotations

from app.config import Settings
from app.core.diversity import dedupe_by_text, mmr_rerank
from app.core.embeddings import EmbeddingService
from app.core.query_rewriter import QueryRewriter
from app.core.reranker import Reranker
from app.core.vector_store import SearchHit, VectorStore


class QueryEngine:
    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingService,
        store: VectorStore,
        reranker: Reranker | None = None,
        rewriter: QueryRewriter | None = None,
    ) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.store = store
        self.reranker = reranker
        self.rewriter = rewriter

    def retrieve(self, query: str, top_k: int, min_score: float) -> list[SearchHit]:
        variants = self._expand(query)

        # A wider pool is needed when a second-stage selector (rerank or MMR)
        # will trim it back down to top_k.
        needs_pool = self.settings.rerank_enabled or self.settings.mmr_enabled
        pool_k = max(self.settings.retrieve_k if needs_pool else top_k, top_k)

        hits = self._multi_query_search(variants, pool_k=pool_k, min_score=min_score)
        if not hits:
            return []

        if self.settings.rerank_enabled and self.reranker:
            hits = self.reranker.rerank(query, hits)

        if self.settings.mmr_enabled and len(hits) > 1:
            vectors = self.store.vectors_for([h.chunk.id for h in hits])
            hits = mmr_rerank(
                hits,
                vectors,
                lambda_=self.settings.mmr_lambda,
                k=top_k,
                dedup_threshold=self.settings.mmr_dedup_threshold,
            )

        if self.settings.text_dedupe_enabled:
            hits = dedupe_by_text(
                hits, jaccard_threshold=self.settings.text_dedupe_jaccard
            )

        return hits[:top_k]

    # ── Stages ───────────────────────────────────────────────
    def _expand(self, query: str) -> list[str]:
        if self.rewriter is None or not self.settings.query_rewrite_enabled:
            return [query]
        return self.rewriter.rewrite(query)

    def _multi_query_search(
        self, variants: list[str], pool_k: int, min_score: float
    ) -> list[SearchHit]:
        """Search each variant and fuse pools, keeping each chunk's best score."""
        query_vecs = self.embeddings.embed_queries(variants)
        fused: dict[str, SearchHit] = {}
        for vec in query_vecs:
            for hit in self.store.search(vec, top_k=pool_k):
                existing = fused.get(hit.chunk.id)
                if existing is None or hit.score > existing.score:
                    fused[hit.chunk.id] = hit

        hits = [h for h in fused.values() if h.score >= min_score]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits
