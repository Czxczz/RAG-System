#!/usr/bin/env python3
"""Run offline RAG evaluation and print aggregate metrics.

This script does NOT call the HTTP /chat endpoint. It invokes the same
in-process pipeline (orchestrator.answer → retrieval + LLM router) that /chat
uses, so metrics reflect real RAG behavior without starting the API server.

Run one case at a time and accumulate results in a report file:

  python scripts/run_eval.py --case-id imdsv2-require --mode ollama --append -v
  python scripts/run_eval.py --case-id insufficient-capacity --mode ollama --append -v
  # … repeat for each case id
"""
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
from app.eval.runner import EvalRunner, build_report
from app.eval.schemas import EvalDataset, EvalReport


def _merge_into_report(
    existing: EvalReport | None,
    new: EvalReport,
    *,
    full_dataset: EvalDataset,
) -> EvalReport:
    """Merge new case results into an existing report (by case_id)."""
    by_id: dict[str, object] = {}
    if existing:
        by_id = {c.case_id: c for c in existing.cases}
    for c in new.cases:
        by_id[c.case_id] = c

    case_order = [c.id for c in full_dataset.cases]
    ordered = [by_id[cid] for cid in case_order if cid in by_id]
    should_refuse = {c.id: c.should_refuse for c in full_dataset.cases}

    return build_report(
        dataset_name=full_dataset.name,
        mode=new.mode,
        top_k=new.top_k,
        cases=ordered,
        should_refuse={c.case_id: should_refuse.get(c.case_id, False) for c in ordered},
    )


def _case_summary(c) -> dict:
    m = c.metrics
    return {
        "case_id": c.case_id,
        "provider": c.provider,
        "grounded": c.grounded,
        "elapsed_seconds": c.elapsed_seconds,
        "precision_at_k": round(m.precision_at_k, 4),
        "recall_at_k": round(m.recall_at_k, 4),
        "redundancy": round(m.redundancy, 4),
        "hallucination": m.hallucination,
        "refused_correctly": m.refused_correctly,
        "notes": c.notes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score RAG quality on a labeled dataset (in-process, not HTTP)."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "eval" / "dataset.ec2.json",
        help="Path to JSON eval dataset.",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        help="LLM mode for answer generation (e.g. ollama, gemini, extractive).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Chunks retrieved per query (default: TOP_K from .env, now 3).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "eval" / "report.ec2.json",
        help="JSON report path (default: eval/report.ec2.json).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Merge this run into an existing --output report (for case-by-case eval).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Pause between cases when running multiple in one invocation.",
    )
    parser.add_argument(
        "--no-rewrite-llm",
        action="store_true",
        help="Disable LLM query rewriting (recommended for Ollama/Gemini eval).",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=None,
        metavar="ID",
        help="Run only this case id (repeat for a few). Omit to run all cases.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print available case ids from the dataset and exit.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-case progress and ETA.",
    )
    args = parser.parse_args()

    full_dataset = EvalDataset.model_validate_json(
        args.dataset.read_text(encoding="utf-8")
    )

    if args.list_cases:
        for c in full_dataset.cases:
            print(f"  {c.id:24s} refuse={c.should_refuse}  {c.query[:50]}")
        return 0

    run_dataset = full_dataset
    if args.case_id:
        wanted = set(args.case_id)
        run_dataset = full_dataset.model_copy(
            update={"cases": [c for c in full_dataset.cases if c.id in wanted]}
        )
        if not run_dataset.cases:
            print(f"No cases matched --case-id {args.case_id}", file=sys.stderr)
            return 1

    settings = get_settings()
    top_k = args.top_k or settings.top_k

    orch = get_orchestrator()
    if orch.registry.count == 0:
        print("Warning: no documents ingested — retrieval metrics will be empty.", file=sys.stderr)

    if args.no_rewrite_llm:
        orch.settings.query_rewrite_use_llm = False
        print("LLM query rewriting disabled (heuristic only).", file=sys.stderr)

    n_run = len(run_dataset.cases)
    if args.mode == "ollama":
        print(
            f"Ollama eval: model={settings.ollama_model} "
            f"timeout={settings.ollama_timeout}s top_k={top_k} cases={n_run}",
            file=sys.stderr,
        )
        est_min = n_run * 60 / 60
        print(f"Estimated wall time: ~{est_min:.0f}–{est_min * 1.5:.0f} min", file=sys.stderr)

    existing: EvalReport | None = None
    if args.append and args.output.exists():
        existing = EvalReport.model_validate_json(args.output.read_text(encoding="utf-8"))
        print(f"Merging into existing report ({len(existing.cases)} cases).", file=sys.stderr)

    def save_partial(result) -> None:
        current_existing: EvalReport | None = None
        if args.output.exists():
            current_existing = EvalReport.model_validate_json(
                args.output.read_text(encoding="utf-8")
            )
        partial = build_report(
            dataset_name=full_dataset.name,
            mode=args.mode,
            top_k=top_k,
            cases=[result],
            should_refuse={
                result.case_id: next(
                    (c.should_refuse for c in full_dataset.cases if c.id == result.case_id),
                    False,
                )
            },
        )
        merged = _merge_into_report(
            current_existing, partial, full_dataset=full_dataset
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(merged.model_dump_json(indent=2), encoding="utf-8")
        if args.verbose:
            print(
                f"  saved → {args.output} ({len(merged.cases)} cases total)",
                file=sys.stderr,
            )

    report = EvalRunner(orch).run_dataset(
        run_dataset,
        default_mode=args.mode,
        default_top_k=top_k,
        sleep_seconds=args.sleep_seconds,
        verbose=args.verbose,
        on_case_complete=save_partial if args.append else None,
    )

    if args.append:
        report = _merge_into_report(existing, report, full_dataset=full_dataset)
    elif existing and not args.case_id:
        report = _merge_into_report(existing, report, full_dataset=full_dataset)

    summary = {
        "dataset": report.dataset,
        "mode": report.mode,
        "top_k": report.top_k,
        "num_cases": report.num_cases,
        "precision_at_k": round(report.precision_at_k, 4),
        "recall_at_k": round(report.recall_at_k, 4),
        "hallucination_rate": round(report.hallucination_rate, 4),
        "citation_accuracy": round(report.citation_accuracy, 4),
        "answer_keyword_recall": round(report.answer_keyword_recall, 4),
        "refusal_accuracy": round(report.refusal_accuracy, 4),
        "redundancy": round(report.redundancy, 4),
        "cases": [_case_summary(c) for c in report.cases],
    }
    print(json.dumps(summary, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nFull report written to {args.output} ({report.num_cases} cases)", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
