"""Query Engine — turns a natural-language query into ranked context chunks.

Pipeline:

    query
      -> rewrite into variants (multi-query)            [query_rewriter]
      -> dense FAISS search (per variant, fused)
      -> BM25 sparse search (optional hybrid)           [bm25_index]
      -> Reciprocal Rank Fusion (dense + BM25)
      -> cosine threshold filter (min_score, dense path)
      -> cross-encoder rerank                            [reranker]
      -> overview / TOC soft demotion (optional)
      -> diversity-aware reranking (MMR)                 [diversity]
      -> top_k

Each stage is independently toggle-able via settings, so the engine degrades
gracefully (e.g. no LLM -> heuristic query rewriting; hybrid off -> dense only).
"""
from __future__ import annotations

from app.config import Settings
from app.core.bm25_index import (
    BM25Index,
    demote_overview_hits,
    reciprocal_rank_fusion,
)
from app.core.diversity import dedupe_by_text, mmr_rerank
from app.core.embeddings import EmbeddingService
from app.core.query_rewriter import QueryRewriter
from app.core.reranker import Reranker
from app.core.spec_retrieval import promote_spec_hits
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
        self._bm25 = BM25Index()
        self._bm25_size = -1

    def retrieve(
        self,
        query: str,
        top_k: int,
        min_score: float,
        document_ids: set[str] | None = None,
    ) -> list[SearchHit]:
        variants = self._expand(query)

        # A wider pool is needed when a second-stage selector (rerank or MMR)
        # will trim it back down to top_k.
        needs_pool = (
            self.settings.rerank_enabled
            or self.settings.mmr_enabled
            or self.settings.hybrid_enabled
        )
        pool_k = max(self.settings.retrieve_k if needs_pool else top_k, top_k)

        dense_hits = self._multi_query_search(
            variants, pool_k=pool_k, min_score=min_score, document_ids=document_ids
        )

        if self.settings.hybrid_enabled:
            hits = self._hybrid_fuse(
                variants,
                dense_hits,
                pool_k=pool_k,
                document_ids=document_ids,
            )
        else:
            hits = dense_hits

        if not hits:
            return []

        if self.settings.rerank_enabled and self.reranker:
            hits = self._rerank_over_variants(variants, hits)

        if self.settings.overview_demote_enabled:
            hits = demote_overview_hits(
                hits, strength=self.settings.overview_demote_strength
            )

        # Keep the full post-rerank pool; MMR/top_k trimming can drop spec tables.
        rerank_pool = hits

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

        hits = promote_spec_hits(query, rerank_pool, top_k)
        return hits[:top_k]

    # ── Stages ───────────────────────────────────────────────
    def _expand(self, query: str) -> list[str]:
        if self.rewriter is None or not self.settings.query_rewrite_enabled:
            return [query]
        return self.rewriter.rewrite(query)

    def _sync_bm25(self) -> None:
        """Rebuild BM25 when the FAISS metadata length changes."""
        n = self.store.num_chunks
        if n == self._bm25_size and self._bm25.size == n:
            return
        # VectorStore metadata order matches FAISS rows.
        self._bm25.rebuild(list(self.store._metadata))  # noqa: SLF001
        self._bm25_size = n

    def _hybrid_fuse(
        self,
        variants: list[str],
        dense_hits: list[SearchHit],
        *,
        pool_k: int,
        document_ids: set[str] | None,
    ) -> list[SearchHit]:
        """Fuse dense + BM25 rankings with Reciprocal Rank Fusion."""
        self._sync_bm25()
        bm25_k = max(self.settings.bm25_top_k, pool_k)
        rankings: list[list[SearchHit]] = []
        if dense_hits:
            rankings.append(dense_hits)

        bm25_fused: dict[str, SearchHit] = {}
        for variant in variants:
            for hit in self._bm25.search(
                variant, top_k=bm25_k, document_ids=document_ids
            ):
                existing = bm25_fused.get(hit.chunk.id)
                if existing is None or hit.score > existing.score:
                    bm25_fused[hit.chunk.id] = hit
        bm25_hits = sorted(
            bm25_fused.values(), key=lambda h: h.score, reverse=True
        )[:bm25_k]
        if bm25_hits:
            rankings.append(bm25_hits)

        if not rankings:
            return []
        if len(rankings) == 1:
            return rankings[0]
        return reciprocal_rank_fusion(rankings, rrf_k=self.settings.rrf_k)

    def _multi_query_search(
        self,
        variants: list[str],
        pool_k: int,
        min_score: float,
        document_ids: set[str] | None = None,
    ) -> list[SearchHit]:
        """Search each variant and fuse pools, keeping each chunk's best score."""
        query_vecs = self.embeddings.embed_queries(variants)
        fused: dict[str, SearchHit] = {}
        for vec in query_vecs:
            for hit in self.store.search(
                vec, top_k=pool_k, document_ids=document_ids
            ):
                existing = fused.get(hit.chunk.id)
                if existing is None or hit.score > existing.score:
                    fused[hit.chunk.id] = hit

        hits = [h for h in fused.values() if h.score >= min_score]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits

    def _rerank_over_variants(
        self, variants: list[str], hits: list[SearchHit]
    ) -> list[SearchHit]:
        """Rerank with each query variant and keep the best score per chunk.

        Multi-query fusion can surface chunks that match a paraphrase but score
        poorly against the original wording. Taking the max cross-encoder score
        across variants aligns reranking with the fused retrieval pool.
        """
        assert self.reranker is not None
        if len(variants) <= 1:
            return self.reranker.rerank(variants[0], hits)

        best: dict[str, SearchHit] = {}
        for variant in variants:
            for hit in self.reranker.rerank(variant, hits):
                existing = best.get(hit.chunk.id)
                if existing is None or hit.score > existing.score:
                    best[hit.chunk.id] = hit
        return sorted(best.values(), key=lambda h: h.score, reverse=True)
