"""Metric helpers for RAG evaluation."""
from __future__ import annotations

import re

from app.core.orchestrator import NOT_FOUND_MESSAGE
from app.core.vector_store import SearchHit
from app.eval.schemas import CaseMetrics, EvalCase

_CITATION_RE = re.compile(r"\[(\d+)\]")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "on", "for", "is", "are",
    "was", "were", "be", "with", "at", "by", "from", "that", "this", "it",
}


def _contains_keyword(text: str, keyword: str) -> bool:
    return keyword.lower() in text.lower()


def _chunk_is_relevant(chunk_text: str, keywords: list[str]) -> bool:
    if not keywords:
        return False
    return any(_contains_keyword(chunk_text, kw) for kw in keywords)


def _context_text(hits: list[SearchHit]) -> str:
    return "\n".join(hit.chunk.text for hit in hits)


def _is_refusal(answer: str) -> bool:
    return answer.strip() == NOT_FOUND_MESSAGE


def precision_at_k(hits: list[SearchHit], case: EvalCase) -> float:
    if not hits:
        return 0.0
    relevant = sum(
        1 for hit in hits if _chunk_is_relevant(hit.chunk.text, case.relevant_keywords)
    )
    return relevant / len(hits)


def recall_at_k(hits: list[SearchHit], case: EvalCase) -> float:
    if not case.relevant_keywords:
        return 1.0 if case.should_refuse else 0.0
    context = _context_text(hits)
    found = sum(1 for kw in case.relevant_keywords if _contains_keyword(context, kw))
    return found / len(case.relevant_keywords)


def answer_keyword_recall(answer: str, case: EvalCase) -> float:
    if not case.expected_answer_keywords:
        return 1.0 if _is_refusal(answer) == case.should_refuse else 0.0
    if _is_refusal(answer):
        return 0.0
    found = sum(
        1 for kw in case.expected_answer_keywords if _contains_keyword(answer, kw)
    )
    return found / len(case.expected_answer_keywords)


def _extract_citation_markers(answer: str) -> list[int]:
    return [int(match) for match in _CITATION_RE.findall(answer)]


def citation_accuracy(answer: str, hits: list[SearchHit]) -> float:
    """Fraction of citation markers that are valid and overlap the cited chunk."""
    if _is_refusal(answer) or not hits:
        return 1.0

    markers = _extract_citation_markers(answer)
    if not markers:
        return 0.0

    supported = 0
    for marker in markers:
        if marker < 1 or marker > len(hits):
            continue
        chunk_text = hits[marker - 1].chunk.text.lower()
        # Claim text immediately before [marker] in the same sentence/line.
        pattern = re.compile(rf"(.+?)\[{marker}\]", re.IGNORECASE | re.DOTALL)
        match = pattern.search(answer)
        if not match:
            continue
        claim = match.group(1).split("\n")[-1]
        tokens = {
            word.lower()
            for word in re.findall(r"[A-Za-z0-9]+", claim)
            if word.lower() not in _STOPWORDS and len(word) > 2
        }
        if not tokens:
            supported += 1
            continue
        overlap = sum(1 for token in tokens if token in chunk_text)
        if overlap / len(tokens) >= 0.25:
            supported += 1

    return supported / len(markers)


def is_hallucination(answer: str, hits: list[SearchHit], case: EvalCase) -> bool:
    """Heuristic hallucination check without an external judge."""
    refused = _is_refusal(answer)

    if case.should_refuse:
        return not refused

    if refused:
        return False

    context = _context_text(hits).lower()
    answer_lower = answer.lower()

    for keyword in case.forbidden_answer_keywords:
        if _contains_keyword(answer_lower, keyword) and not _contains_keyword(context, keyword):
            return True

    for number in _NUMBER_RE.findall(answer):
        if number not in context:
            return True

    return False


def score_case(
    case: EvalCase,
    answer: str,
    hits: list[SearchHit],
) -> CaseMetrics:
    refused = _is_refusal(answer)
    retrieved_relevant = sum(
        1 for hit in hits if _chunk_is_relevant(hit.chunk.text, case.relevant_keywords)
    )

    return CaseMetrics(
        precision_at_k=precision_at_k(hits, case),
        recall_at_k=recall_at_k(hits, case),
        hallucination=is_hallucination(answer, hits, case),
        citation_accuracy=citation_accuracy(answer, hits),
        answer_keyword_recall=answer_keyword_recall(answer, case),
        refused_correctly=refused if case.should_refuse else not refused,
        retrieved_relevant=retrieved_relevant,
        retrieved_total=len(hits),
    )
