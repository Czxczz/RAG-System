"""Cross-encoder reranker for second-stage retrieval scoring.

After bi-encoder (embedding) search returns a candidate pool, a cross-encoder
scores each (query, chunk) pair jointly and re-orders by relevance.
"""
from __future__ import annotations

from functools import lru_cache

from app.config import Settings, get_settings
from app.core.vector_store import SearchHit


class Reranker:
    """Lazy-loading cross-encoder backend."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.settings.rerank_model)
        return self._model

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        if not hits:
            return hits
        model = self._load()
        pairs = [(query, hit.chunk.text) for hit in hits]
        scores = model.predict(pairs, show_progress_bar=False)
        scored = sorted(
            zip(hits, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [SearchHit(chunk=hit.chunk, score=float(score)) for hit, score in scored]


@lru_cache
def get_reranker() -> Reranker:
    return Reranker(get_settings())
