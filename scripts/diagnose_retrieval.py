#!/usr/bin/env python3
"""Evidence-gathering diagnostic for retrieval quality.

Probes the *live* index to decide whether query rewriting and/or hybrid
(BM25 + dense) retrieval are warranted. Pure retrieval — no LLM answer calls
(LLM is only used, optionally, to generate query-rewrite variants).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from app.core.diversity import mmr_rerank
from app.dependencies import get_orchestrator

orch = get_orchestrator()
S = orch.settings
store = orch.store
emb = orch.embeddings
meta = store._metadata  # diagnostic access to chunk texts

print(f"corpus: {store.num_chunks} chunks | TOP_K={S.top_k} "
      f"RETRIEVE_K={S.retrieve_k} MIN_SCORE={S.min_score} "
      f"rerank={S.rerank_enabled} mmr={S.mmr_enabled} lambda={S.mmr_lambda}")

# ── Lexical presence of candidate exact-match terms ──────────
TERMS = [
    "IMDSv2", "InsufficientInstanceCapacity", "RunInstances", "t2.micro",
    "DeleteOnTermination", "Elastic IP", "placement group", "spot instance",
    "user data", "security group",
]
lower_texts = [c.text.lower() for c in meta]
print("\n=== lexical term presence in corpus ===")
present = {}
for t in TERMS:
    cnt = sum(1 for x in lower_texts if t.lower() in x)
    present[t] = cnt
    print(f"  {t:32s} {cnt:5d} chunks")


def keyword_rows(keyword: str) -> set[int]:
    kw = keyword.lower()
    return {i for i, x in enumerate(lower_texts) if kw in x}


def dense_rank_of_keyword(query: str, keyword: str, depth: int = 1000) -> int | None:
    """Rank (1-based) of the first keyword-bearing chunk in dense results."""
    rows = keyword_rows(keyword)
    if not rows:
        return None
    qv = emb.embed_query(query)
    hits = store.search(qv, top_k=depth)
    id_to_row = {c.id: i for i, c in enumerate(meta)}
    for rank, h in enumerate(hits, start=1):
        if id_to_row[h.chunk.id] in rows:
            return rank
    return -1  # exists in corpus but not in top `depth`


def redundancy(hits) -> float:
    """Max pairwise cosine among final hits (1.0 == exact duplicate)."""
    if len(hits) < 2:
        return 0.0
    vecs = store.vectors_for([h.chunk.id for h in hits])
    m = np.vstack([vecs[h.chunk.id] for h in hits])
    sim = m @ m.T
    np.fill_diagonal(sim, -1)
    return float(sim.max())


def pipeline(query: str, variants: list[str]):
    pool = orch.query_engine._multi_query_search(
        variants, pool_k=S.retrieve_k, min_score=S.min_score
    )
    rr = orch.query_engine.reranker
    reranked = rr.rerank(query, pool) if (S.rerank_enabled and rr) else pool
    if S.mmr_enabled and len(reranked) > S.top_k:
        vecs = store.vectors_for([h.chunk.id for h in reranked])
        final = mmr_rerank(reranked, vecs, lambda_=S.mmr_lambda, k=S.top_k)
    else:
        final = reranked[:S.top_k]
    return pool, final


def kw_hit(hits, keyword: str | None) -> str:
    if not keyword:
        return "n/a"
    return str(sum(1 for h in hits if keyword.lower() in h.chunk.text.lower()))


# query, probe keyword (None for purely conceptual), kind
QUERIES = [
    ("How do I let my instance receive traffic from the internet?", "security group", "conceptual"),
    ("How can I keep my data after I terminate an instance?", "DeleteOnTermination", "conceptual"),
    ("How do I require IMDSv2 on my instances?", "IMDSv2", "identifier"),
    ("What causes an InsufficientInstanceCapacity error?", "InsufficientInstanceCapacity", "identifier"),
    ("What is the request rate limit for the RunInstances API?", "RunInstances", "identifier"),
    ("Is the t2.micro eligible for the free tier?", "t2.micro", "identifier"),
]

for query, kw, kind in QUERIES:
    print("\n" + "=" * 78)
    print(f"[{kind}] {query!r}  (probe='{kw}')")

    # rewriting OFF
    pool_off, final_off = pipeline(query, [query])
    # rewriting ON (LLM if available, else heuristic)
    variants = orch.rewriter.rewrite(query)
    pool_on, final_on = pipeline(query, variants)

    drank = dense_rank_of_keyword(query, kw) if kw else None

    print(f"  variants (rewrite ON): {variants}")
    print(f"  dense rank of first '{kw}' chunk: {drank}  "
          f"(<= RETRIEVE_K={S.retrieve_k} means reranker can see it)")
    print(f"  rewrite OFF: pool={len(pool_off):2d} kw_in_final={kw_hit(final_off, kw)} "
          f"redundancy_max={redundancy(final_off):.3f} "
          f"top_scores={[round(h.score,2) for h in final_off]}")
    print(f"  rewrite ON : pool={len(pool_on):2d} kw_in_final={kw_hit(final_on, kw)} "
          f"redundancy_max={redundancy(final_on):.3f} "
          f"top_scores={[round(h.score,2) for h in final_on]}")
    off_ids = [h.chunk.id for h in final_off]
    on_ids = [h.chunk.id for h in final_on]
    changed = off_ids != on_ids
    print(f"  final set changed by rewriting: {changed} "
          f"(new chunks: {len(set(on_ids) - set(off_ids))})")
