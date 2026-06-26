"""Run labeled evaluation cases against the live RAG stack."""
from __future__ import annotations

from app.core.orchestrator import RAGOrchestrator
from app.eval.metrics import score_case
from app.eval.schemas import CaseResult, EvalCase, EvalDataset, EvalReport


def _aggregate(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


class EvalRunner:
    def __init__(self, orchestrator: RAGOrchestrator) -> None:
        self.orch = orchestrator

    def run_case(self, case: EvalCase, default_top_k: int) -> CaseResult:
        top_k = case.top_k or default_top_k
        result = self.orch.answer(query=case.query, mode=case.mode, top_k=top_k)
        metrics = score_case(case, result.answer, result.hits)

        notes: list[str] = []
        if case.should_refuse and not metrics.refused_correctly:
            notes.append("Expected refusal but got an answer.")
        if not case.should_refuse and metrics.recall_at_k < 1.0:
            notes.append("Not all expected retrieval keywords found in top-k.")
        if metrics.hallucination:
            notes.append("Hallucination heuristic triggered.")

        return CaseResult(
            case_id=case.id,
            query=case.query,
            answer=result.answer,
            provider=result.provider,
            grounded=result.grounded,
            metrics=metrics,
            notes=notes,
        )

    def run_dataset(
        self,
        dataset: EvalDataset,
        *,
        default_mode: str = "auto",
        default_top_k: int = 5,
    ) -> EvalReport:
        cases: list[CaseResult] = []
        for case in dataset.cases:
            if case.mode == "auto" and default_mode != "auto":
                case = case.model_copy(update={"mode": default_mode})
            cases.append(self.run_case(case, default_top_k=default_top_k))

        refusal_cases = [c for c in dataset.cases if c.should_refuse]
        refusal_results = [r for r, c in zip(cases, dataset.cases) if c.should_refuse]

        return EvalReport(
            dataset=dataset.name,
            mode=default_mode,
            top_k=default_top_k,
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
            if refusal_cases
            else 1.0,
            cases=cases,
        )
