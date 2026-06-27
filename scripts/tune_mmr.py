#!/usr/bin/env python3
"""Sweep MMR (lambda, dedup_threshold) and report retrieval-only metrics.

Loads the index once and re-runs retrieval per case for each parameter combo
(no LLM calls), so it's fast and isolates the effect of MMR on precision,
recall, and redundancy.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dependencies import get_orchestrator
from app.eval.metrics import precision_at_k, recall_at_k, redundancy
from app.eval.schemas import EvalDataset

orch = get_orchestrator()
S = orch.settings
dataset = EvalDataset.model_validate_json(
    (ROOT / "eval" / "dataset.ec2.json").read_text(encoding="utf-8")
)
answerable = [c for c in dataset.cases if not c.should_refuse]

COMBOS = [
    (0.7, 1.01),   # baseline: old behaviour (no dedup)
    (0.7, 0.95),
    (0.7, 0.90),
    (0.7, 0.85),
    (0.5, 0.90),
    (0.5, 0.85),
    (0.6, 0.88),
]

print(f"answerable cases: {len(answerable)} | TOP_K={S.top_k} MIN_SCORE={S.min_score}")
print(f"{'lambda':>7} {'dedup':>6} | {'precision':>9} {'recall':>7} {'redundancy':>10} {'avg_hits':>8}")
print("-" * 60)

for lam, thr in COMBOS:
    S.mmr_lambda = lam
    S.mmr_dedup_threshold = thr
    precisions, recalls, reds, counts = [], [], [], []
    for case in answerable:
        top_k = case.top_k or S.top_k
        hits = orch.query_engine.retrieve(case.query, top_k=top_k, min_score=S.min_score)
        vecs = orch.store.vectors_for([h.chunk.id for h in hits])
        precisions.append(precision_at_k(hits, case))
        recalls.append(recall_at_k(hits, case))
        reds.append(redundancy(hits, vecs))
        counts.append(len(hits))

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print(f"{lam:>7.2f} {thr:>6.2f} | {avg(precisions):>9.3f} {avg(recalls):>7.3f} "
          f"{avg(reds):>10.3f} {avg(counts):>8.2f}")
