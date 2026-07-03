"""Run labeled evaluation cases against the live RAG stack."""
from __future__ import annotations

import sys
import time
from collections.abc import Callable

from app.eval.corpus import resolve_document_ids
from app.core.orchestrator import RAGOrchestrator
from app.eval.metrics import redundancy, score_case
from app.eval.schemas import CaseResult, EvalCase, EvalDataset, EvalReport


def _aggregate(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def build_report(
    *,
    dataset_name: str,
    mode: str,
    top_k: int,
    cases: list[CaseResult],
    should_refuse: dict[str, bool],
) -> EvalReport:
    """Build an EvalReport from case results and recompute aggregate metrics."""
    refusal_ids = {cid for cid, refuse in should_refuse.items() if refuse}
    refusal_results = [c for c in cases if c.case_id in refusal_ids]

    return EvalReport(
        dataset=dataset_name,
        mode=mode,
        top_k=top_k,
        num_cases=len(cases),
        precision_at_k=_aggregate([c.metrics.precision_at_k for c in cases]),
        recall_at_k=_aggregate([c.metrics.recall_at_k for c in cases]),
        hallucination_rate=_aggregate(
            [1.0 if c.metrics.hallucination else 0.0 for c in cases]
        ),
        citation_accuracy=_aggregate([c.metrics.citation_accuracy for c in cases]),
        answer_keyword_recall=_aggregate(
            [c.metrics.answer_keyword_recall for c in cases]
        ),
        refusal_accuracy=_aggregate(
            [1.0 if r.metrics.refused_correctly else 0.0 for r in refusal_results]
        )
        if refusal_ids
        else 1.0,
        source_accuracy=_aggregate([c.metrics.source_accuracy for c in cases]),
        redundancy=_aggregate([c.metrics.redundancy for c in cases]),
        cases=cases,
    )


class EvalRunner:
    def __init__(self, orchestrator: RAGOrchestrator) -> None:
        self.orch = orchestrator

    def run_case(self, case: EvalCase, default_top_k: int) -> CaseResult:
        top_k = case.top_k or default_top_k
        document_ids = resolve_document_ids(self.orch.registry, case.document_filenames)
        t0 = time.monotonic()
        result = self.orch.answer(
            query=case.query,
            mode=case.mode,
            top_k=top_k,
            document_ids=document_ids,
        )
        elapsed = time.monotonic() - t0
        vectors = self.orch.store.vectors_for([h.chunk.id for h in result.hits])
        redundancy_score = redundancy(result.hits, vectors)
        metrics = score_case(
            case, result.answer, result.hits, redundancy_score=redundancy_score
        )

        notes: list[str] = []
        if case.should_refuse and not metrics.refused_correctly:
            notes.append("Expected refusal but got an answer.")
        if not case.should_refuse and metrics.recall_at_k < 1.0:
            notes.append("Not all expected retrieval keywords found in top-k.")
        if metrics.hallucination:
            notes.append("Hallucination heuristic triggered.")
        if case.expected_source_filenames and metrics.source_accuracy < 1.0:
            notes.append("Expected source document not found in top-k.")

        return CaseResult(
            case_id=case.id,
            query=case.query,
            answer=result.answer,
            provider=result.provider,
            grounded=result.grounded,
            metrics=metrics,
            notes=notes,
            elapsed_seconds=round(elapsed, 1),
        )

    def run_dataset(
        self,
        dataset: EvalDataset,
        *,
        default_mode: str = "auto",
        default_top_k: int = 3,
        sleep_seconds: float = 0.0,
        verbose: bool = False,
        on_case_complete: Callable[[CaseResult], None] | None = None,
    ) -> EvalReport:
        cases: list[CaseResult] = []
        total = len(dataset.cases)
        t0 = time.monotonic()
        should_refuse = {c.id: c.should_refuse for c in dataset.cases}

        for i, case in enumerate(dataset.cases):
            if i > 0 and sleep_seconds > 0:
                if verbose:
                    print(f"  sleeping {sleep_seconds}s…", file=sys.stderr)
                time.sleep(sleep_seconds)

            if case.mode == "auto" and default_mode != "auto":
                case = case.model_copy(update={"mode": default_mode})

            if verbose:
                print(
                    f"[{i + 1}/{total}] {case.id}: {case.query[:60]}…",
                    file=sys.stderr,
                    flush=True,
                )

            result = self.run_case(case, default_top_k=default_top_k)
            cases.append(result)

            if verbose:
                m = result.metrics
                print(
                    f"  done in {result.elapsed_seconds:.0f}s | provider={result.provider} "
                    f"P={m.precision_at_k:.2f} R={m.recall_at_k:.2f} "
                    f"red={m.redundancy:.2f}",
                    file=sys.stderr,
                    flush=True,
                )
                if i + 1 < total:
                    avg = (time.monotonic() - t0) / (i + 1)
                    eta = avg * (total - i - 1)
                    print(f"  ETA ~{eta / 60:.1f} min remaining", file=sys.stderr, flush=True)

            if on_case_complete:
                on_case_complete(result)

        return build_report(
            dataset_name=dataset.name,
            mode=default_mode,
            top_k=default_top_k,
            cases=cases,
            should_refuse=should_refuse,
        )
