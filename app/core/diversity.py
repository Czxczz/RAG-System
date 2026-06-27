"""Diversity-aware reranking via Maximal Marginal Relevance (MMR).

After relevance ranking (cross-encoder or cosine), candidate chunks are often
near-duplicates — they repeat the same fact and waste the context window. MMR
greedily selects the next chunk that maximises:

    score(c) = lambda * relevance(c) - (1 - lambda) * max_sim(c, already_selected)

so each pick is relevant *and* adds new information. lambda=1.0 reduces to pure
relevance; lambda=0.0 to pure diversity.
"""
from __future__ import annotations

import re

import numpy as np

from app.core.vector_store import SearchHit


def _minmax_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def mmr_order(
    relevance: list[float],
    similarity: np.ndarray,
    lambda_: float,
    k: int,
    dedup_threshold: float = 1.01,
) -> list[int]:
    """Return indices selected by MMR, in selection order.

    `similarity` is an (n, n) matrix of pairwise similarities (cosine for
    normalised vectors). `relevance` is a length-n list of relevance scores.

    `dedup_threshold` is a hard redundancy cap: a candidate whose similarity to
    any already-selected item is >= the threshold is skipped entirely. This
    removes near/exact duplicates that the soft MMR penalty would otherwise
    keep when their relevance is high. The default (>1.0) disables it.
    """
    n = len(relevance)
    k = min(k, n)
    if n == 0 or k == 0:
        return []

    rel = _minmax_normalize(relevance)
    selected: list[int] = []
    remaining = set(range(n))

    while remaining and len(selected) < k:
        best_idx = None
        best_score = -float("inf")
        for i in remaining:
            redundancy = max((similarity[i][j] for j in selected), default=0.0)
            if redundancy >= dedup_threshold:
                continue  # near-duplicate of an already-selected chunk
            score = lambda_ * rel[i] - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx is None:
            break  # only duplicates remain -> stop rather than pad with dupes
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


def mmr_rerank(
    hits: list[SearchHit],
    vectors_by_id: dict[str, np.ndarray],
    lambda_: float,
    k: int,
    dedup_threshold: float = 1.01,
) -> list[SearchHit]:
    """Reorder/trim `hits` to the top-`k` most relevant *and* diverse.

    Hits whose vectors are unavailable fall back to identity ordering. With a
    `dedup_threshold` < 1.0, the result may contain fewer than `k` hits when
    the remaining candidates are all near-duplicates (this is intentional —
    duplicates only waste the context window).
    """
    if len(hits) <= 1:
        return hits[:k]

    try:
        matrix = np.vstack([vectors_by_id[h.chunk.id] for h in hits]).astype("float32")
    except KeyError:
        return hits[:k]  # missing a vector -> don't risk a bad reorder

    # Vectors are L2-normalised at indexing time, so dot product == cosine.
    similarity = matrix @ matrix.T
    relevance = [h.score for h in hits]
    order = mmr_order(
        relevance, similarity, lambda_=lambda_, k=k, dedup_threshold=dedup_threshold
    )
    return [hits[i] for i in order]


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(cleaned.split())


def _word_jaccard(a: str, b: str) -> float:
    wa, wb = set(_normalize_text(a).split()), set(_normalize_text(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def dedupe_by_text(
    hits: list[SearchHit],
    *,
    jaccard_threshold: float = 0.85,
) -> list[SearchHit]:
    """Drop later chunks whose text closely repeats an earlier kept chunk.

    Complements MMR (embedding space) by catching exact boilerplate and
    copy-paste passages that embeddings treat as slightly distinct.
    """
    if len(hits) <= 1:
        return hits

    kept: list[SearchHit] = []
    seen_normalized: set[str] = set()
    for hit in hits:
        norm = _normalize_text(hit.chunk.text)
        if norm in seen_normalized:
            continue
        if any(_word_jaccard(hit.chunk.text, prev.chunk.text) >= jaccard_threshold for prev in kept):
            continue
        kept.append(hit)
        seen_normalized.add(norm)
    return kept
