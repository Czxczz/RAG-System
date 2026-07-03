"""Unit tests for RAG evaluation metrics."""
from __future__ import annotations

from app.core.orchestrator import NOT_FOUND_MESSAGE
from app.core.vector_store import SearchHit, StoredChunk
from app.eval.metrics import (
    citation_accuracy,
    is_hallucination,
    precision_at_k,
    recall_at_k,
    source_accuracy,
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


def test_citation_markers_not_flagged_as_invented_numbers():
    # Markers [1]..[4] must not be treated as fabricated numeric facts when the
    # underlying chunk text contains no such numbers.
    case = EvalCase(
        id="eip",
        query="What is an Elastic IP address?",
        relevant_keywords=["Elastic IP"],
        should_refuse=False,
    )
    hits = [_hit("An Elastic IP address is a static public address.", 0)]
    answer = "It is static [1]. It is reachable from the internet [2][3][4]."
    assert is_hallucination(answer, hits, case) is False


def test_year_in_answer_is_not_hallucination():
    case = EvalCase(
        id="free-tier",
        query="Is t2.micro free tier eligible?",
        relevant_keywords=["free tier"],
        should_refuse=False,
    )
    hits = [_hit("t2.micro is free tier eligible for new accounts.", 0)]
    answer = "Yes, if your account was created before July 15, 2025 [1]."
    # 15 is not in context, so it still flags — but a pure-year answer should not.
    year_only = "Eligibility started in 2025 [1]."
    assert is_hallucination(year_only, hits, case) is False


def test_invented_number_still_flagged():
    case = EvalCase(
        id="leave",
        query="How many annual leave days?",
        relevant_keywords=["14 days"],
        should_refuse=False,
    )
    hits = [_hit("annual leave policy details without numbers", 0)]
    answer = "Employees receive 99 days of leave [1]."
    assert is_hallucination(answer, hits, case) is True


def test_source_accuracy_requires_expected_filename():
    case = EvalCase(
        id="types",
        query="How many vCPUs does m5.xlarge have?",
        relevant_keywords=["m5.xlarge", "vCPU"],
        expected_source_filenames=["ec2-types.pdf"],
    )
    wrong_doc = SearchHit(
        chunk=StoredChunk(
            id="ug:0",
            document_id="ug",
            filename="ec2-ug.pdf",
            chunk_index=0,
            page=1,
            text="general EC2 overview",
        ),
        score=0.8,
    )
    right_doc = SearchHit(
        chunk=StoredChunk(
            id="types:0",
            document_id="types",
            filename="ec2-types.pdf",
            chunk_index=0,
            page=1,
            text="m5.xlarge provides 4 vCPU",
        ),
        score=0.7,
    )
    assert source_accuracy([wrong_doc], case) == 0.0
    assert source_accuracy([right_doc], case) == 1.0
