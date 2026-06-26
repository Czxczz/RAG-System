"""Unit tests for RAG evaluation metrics."""
from __future__ import annotations

from app.core.orchestrator import NOT_FOUND_MESSAGE
from app.core.vector_store import SearchHit, StoredChunk
from app.eval.metrics import (
    citation_accuracy,
    is_hallucination,
    precision_at_k,
    recall_at_k,
)
from app.eval.schemas import EvalCase


def _hit(text: str, chunk_index: int = 0) -> SearchHit:
    return SearchHit(
        chunk=StoredChunk(
            id=f"doc:{chunk_index}",
            document_id="doc",
            filename="handbook.pdf",
            chunk_index=chunk_index,
            page=1,
            text=text,
        ),
        score=0.8,
    )


def test_precision_and_recall_at_k():
    case = EvalCase(
        id="leave",
        query="How many annual leave days?",
        relevant_keywords=["14 days", "18 days"],
    )
    hits = [
        _hit("annual leave entitlement 14 days for staff", 0),
        _hit("mileage claims policy", 1),
    ]

    assert precision_at_k(hits, case) == 0.5
    assert recall_at_k(hits, case) == 0.5


def test_refusal_case_not_hallucination():
    case = EvalCase(
        id="remote",
        query="Can employees work remotely?",
        relevant_keywords=["remote"],
        should_refuse=True,
    )
    answer = NOT_FOUND_MESSAGE
    assert is_hallucination(answer, [], case) is False


def test_hallucination_when_answer_invents_number():
    case = EvalCase(
        id="leave",
        query="How many annual leave days?",
        relevant_keywords=["14 days"],
        should_refuse=False,
    )
    hits = [_hit("annual leave policy details without numbers", 0)]
    answer = "Employees receive 99 days of leave [1]."
    assert is_hallucination(answer, hits, case) is True


def test_citation_accuracy_valid_marker():
    hits = [_hit("annual leave 14 days 16 days 18 days", 0)]
    answer = "Employees get 14 days of annual leave [1]."
    assert citation_accuracy(answer, hits) == 1.0
