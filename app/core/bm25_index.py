"""Sparse BM25 lexical retrieval + fusion helpers for hybrid RAG.

Dense embeddings miss rare exact tokens (API symbols, error codes, product
ids). BM25 ranks by term frequency so those chunks enter the candidate pool
before cross-encoder rerank.

No third-party dependency: Okapi BM25 is implemented here.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from app.core.vector_store import SearchHit, StoredChunk

_TOKEN_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_\.\+\-]{1,63}")

# Soft demotion for chapter intros / TOC / wrap-ups that steal top ranks from
# instructional content. Patterns are deliberately general (books + manuals).
_OVERVIEW_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"by the end of this (chapter|section|module)",
        r"in this chapter[, ]+we (will |('ll )?|learned|covered|discussed)",
        r"in this section[, ]+we('ll| will)",
        r"the following topics",
        r"table of contents",
        r"^\s*summary\s*$",
        r"in (the )?next (chapter|section)",
        r"we (have )?successfully (built|learned|implemented|secured)",
    )
)
_TOC_DOTS_RE = re.compile(r"(?:\.\s*){6,}")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric / code-ish tokens for BM25."""
    if not text:
        return []
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


class BM25Index:
    """In-memory Okapi BM25 index over chunk texts.

    Rebuild from the vector-store metadata whenever the chunk count changes
    (lazy sync in ``QueryEngine``).
    """

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[StoredChunk] = []
        self._docs: list[list[str]] = []
        self._doc_len: list[int] = []
        self._avgdl: float = 0.0
        self._df: Counter[str] = Counter()
        self._idf: dict[str, float] = {}
        self._n: int = 0

    @property
    def size(self) -> int:
        return self._n

    def rebuild(self, chunks: list[StoredChunk]) -> None:
        self._chunks = list(chunks)
        self._docs = [tokenize(c.text) for c in self._chunks]
        self._doc_len = [len(d) for d in self._docs]
        self._n = len(self._docs)
        self._avgdl = (sum(self._doc_len) / self._n) if self._n else 0.0
        self._df = Counter()
        for doc in self._docs:
            for term in set(doc):
                self._df[term] += 1
        self._idf = {}
        for term, df in self._df.items():
            # Standard BM25 idf with +0.5 smoothing.
            self._idf[term] = math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))

    def search(
        self,
        query: str,
        top_k: int,
        document_ids: set[str] | None = None,
    ) -> list[SearchHit]:
        if self._n == 0 or top_k <= 0:
            return []
        q_terms = tokenize(query)
        if not q_terms:
            return []

        scores: list[tuple[float, int]] = []
        for i, doc in enumerate(self._docs):
            chunk = self._chunks[i]
            if document_ids is not None and chunk.document_id not in document_ids:
                continue
            tf = Counter(doc)
            score = 0.0
            dl = self._doc_len[i] or 1
            for term in q_terms:
                if term not in tf:
                    continue
                idf = self._idf.get(term, 0.0)
                freq = tf[term]
                denom = freq + self.k1 * (1.0 - self.b + self.b * dl / max(self._avgdl, 1e-9))
                score += idf * (freq * (self.k1 + 1.0)) / denom
            if score > 0.0:
                scores.append((score, i))

        scores.sort(key=lambda x: x[0], reverse=True)
        hits: list[SearchHit] = []
        for score, idx in scores[:top_k]:
            hits.append(SearchHit(chunk=self._chunks[idx], score=float(score)))
        return hits


def reciprocal_rank_fusion(
    rankings: list[list[SearchHit]],
    *,
    rrf_k: int = 60,
) -> list[SearchHit]:
    """Merge ranked lists with Reciprocal Rank Fusion.

    ``score(d) = Σ 1 / (rrf_k + rank_i(d))`` across input rankings.
    Preserves each chunk's identity; the fused score replaces prior scores.
    """
    if not rankings:
        return []
    fused: dict[str, float] = {}
    chunks: dict[str, StoredChunk] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            cid = hit.chunk.id
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            chunks[cid] = hit.chunk
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [SearchHit(chunk=chunks[cid], score=score) for cid, score in ordered]


def overview_penalty(text: str) -> float:
    """Return 0..1 penalty for overview / TOC / wrap-up style passages."""
    if not text:
        return 0.0
    penalty = 0.0
    lower = text.lower()
    matches = sum(1 for p in _OVERVIEW_PATTERNS if p.search(text))
    if matches:
        penalty += min(0.45, 0.18 * matches)
    if _TOC_DOTS_RE.search(text):
        penalty += 0.25
    # Dense dotted leaders often mean TOC rows.
    if lower.count(". . .") >= 3 or lower.count("....") >= 2:
        penalty += 0.15
    return min(1.0, penalty)


def demote_overview_hits(
    hits: list[SearchHit],
    *,
    strength: float = 0.55,
) -> list[SearchHit]:
    """Down-weight overview-like chunks, then re-sort by score.

    ``strength`` scales how hard the penalty hits (0 = off, 1 = full).
    """
    if not hits or strength <= 0:
        return hits
    adjusted: list[SearchHit] = []
    for hit in hits:
        pen = overview_penalty(hit.chunk.text) * strength
        new_score = hit.score * (1.0 - pen)
        adjusted.append(SearchHit(chunk=hit.chunk, score=new_score))
    adjusted.sort(key=lambda h: h.score, reverse=True)
    return adjusted
