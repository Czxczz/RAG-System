#!/usr/bin/env python3
"""Run offline RAG evaluation and print aggregate metrics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.dependencies import get_orchestrator
from app.eval.runner import EvalRunner
from app.eval.schemas import EvalDataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Score RAG quality on a labeled dataset.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "eval" / "dataset.example.json",
        help="Path to JSON eval dataset.",
    )
    parser.add_argument("--mode", default="auto", help="LLM mode passed to /chat pipeline.")
    parser.add_argument("--top-k", type=int, default=None, help="Default top_k for cases.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write full JSON report.",
    )
    args = parser.parse_args()

    dataset = EvalDataset.model_validate_json(args.dataset.read_text(encoding="utf-8"))
    settings = get_settings()
    top_k = args.top_k or settings.top_k

    orch = get_orchestrator()
    if orch.registry.count == 0:
        print("Warning: no documents ingested — retrieval metrics will be empty.", file=sys.stderr)

    report = EvalRunner(orch).run_dataset(
        dataset,
        default_mode=args.mode,
        default_top_k=top_k,
    )

    summary = {
        "dataset": report.dataset,
        "num_cases": report.num_cases,
        "precision_at_k": round(report.precision_at_k, 4),
        "recall_at_k": round(report.recall_at_k, 4),
        "hallucination_rate": round(report.hallucination_rate, 4),
        "citation_accuracy": round(report.citation_accuracy, 4),
        "answer_keyword_recall": round(report.answer_keyword_recall, 4),
        "refusal_accuracy": round(report.refusal_accuracy, 4),
    }
    print(json.dumps(summary, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nFull report written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
