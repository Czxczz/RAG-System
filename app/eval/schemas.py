"""Evaluation dataset and result schemas."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """One labeled question for offline evaluation."""

    id: str
    query: str
    # Keywords that mark a retrieved chunk as relevant (case-insensitive).
    relevant_keywords: list[str] = Field(default_factory=list)
    # Keywords expected in a good grounded answer.
    expected_answer_keywords: list[str] = Field(default_factory=list)
    # Keywords that should NOT appear unless present in retrieved context.
    forbidden_answer_keywords: list[str] = Field(default_factory=list)
    # True when the system should refuse (topic not in corpus).
    should_refuse: bool = False
    # Restrict retrieval to these ingested filenames (omit = search all docs).
    document_filenames: list[str] = Field(default_factory=list)
    # At least one retrieved chunk should come from one of these filenames.
    expected_source_filenames: list[str] = Field(default_factory=list)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    mode: Literal["auto", "openai", "gemini", "ollama", "extractive"] = "auto"


class EvalDataset(BaseModel):
    name: str = "rag-eval"
    cases: list[EvalCase]


class CaseMetrics(BaseModel):
    precision_at_k: float
    recall_at_k: float
    hallucination: bool
    citation_accuracy: float
    answer_keyword_recall: float
    refused_correctly: bool
    retrieved_relevant: int
    retrieved_total: int
    source_accuracy: float = 1.0
    # Max pairwise cosine among the final retrieved chunks (0 == none/single,
    # 1.0 == an exact duplicate). Lower is better.
    redundancy: float = 0.0


class CaseResult(BaseModel):
    case_id: str
    query: str
    answer: str
    provider: str
    grounded: bool
    metrics: CaseMetrics
    notes: list[str] = Field(default_factory=list)
    elapsed_seconds: float = 0.0


class EvalReport(BaseModel):
    dataset: str
    mode: str
    top_k: int
    num_cases: int
    precision_at_k: float
    recall_at_k: float
    hallucination_rate: float
    citation_accuracy: float
    answer_keyword_recall: float
    refusal_accuracy: float
    source_accuracy: float = 1.0
    redundancy: float = 0.0
    cases: list[CaseResult]
