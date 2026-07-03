"""Metric helpers for RAG evaluation."""
from __future__ import annotations

import re

from app.core.orchestrator import NOT_FOUND_MESSAGE
from app.core.vector_store import SearchHit
from app.eval.schemas import CaseMetrics, EvalCase

_CITATION_RE = re.compile(r"\[(\d+)\]")
# Matches a whole citation group, e.g. "[1]", "[2, 4]", "[1,3]" — stripped before
# number-based hallucination checks so citation markers aren't mistaken for facts.
_CITATION_GROUP_RE = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")
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
    markers: list[int] = []
    for group in _CITATION_GROUP_RE.findall(answer):
        markers.extend(int(part.strip()) for part in group.split(","))
    return markers


def _citation_groups(answer: str) -> list[tuple[str, list[int]]]:
    pattern = re.compile(
        r"(.+?)\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]", re.IGNORECASE | re.DOTALL
    )
    groups: list[tuple[str, list[int]]] = []
    for match in pattern.finditer(answer):
        claim = match.group(1).split("\n")[-1]
        markers = [int(part.strip()) for part in match.group(2).split(",")]
        groups.append((claim, markers))
    return groups


def citation_accuracy(answer: str, hits: list[SearchHit]) -> float:
    """Fraction of citation groups that are valid and overlap a cited chunk."""
    if _is_refusal(answer) or not hits:
        return 1.0

    groups = _citation_groups(answer)
    if not groups:
        return 0.0

    supported = 0
    for claim, markers in groups:
        valid = [m for m in markers if 1 <= m <= len(hits)]
        if not valid:
            continue
        tokens = {
            word.lower()
            for word in re.findall(r"[A-Za-z0-9]+", claim)
            if word.lower() not in _STOPWORDS and len(word) > 2
        }
        if not tokens:
            supported += 1
            continue
        backed = False
        for marker in valid:
            chunk_text = hits[marker - 1].chunk.text.lower()
            overlap = sum(1 for token in tokens if token in chunk_text)
            if overlap / len(tokens) >= 0.25:
                backed = True
                break
        if backed:
            supported += 1

    return supported / len(groups)


def _strip_citations(text: str) -> str:
    """Remove inline citation markers so they aren't parsed as factual numbers."""
    return _CITATION_GROUP_RE.sub(" ", text)


def _number_tokens(text: str) -> set[str]:
    return set(_NUMBER_RE.findall(text))


def _is_year(token: str) -> bool:
    return token.isdigit() and len(token) == 4 and "1900" <= token <= "2099"


def is_hallucination(answer: str, hits: list[SearchHit], case: EvalCase) -> bool:
    """Heuristic hallucination check without an external judge.

    Two signals, both designed to minimise false positives:

    * a *forbidden* keyword appears in the answer but not in the retrieved
      context (an explicit, per-case red flag), or
    * the answer states a numeric fact that appears nowhere in the retrieved
      context. Citation markers (``[1]``, ``[2, 4]``) are stripped first so they
      are not mistaken for facts, and 4-digit years are ignored (they are rarely
      fabricated and often reflect formatting differences).

    Numbers are compared as whole tokens against the set of numbers in context,
    rather than substring matching, so ``14`` is not silently "found" inside
    ``2014``.
    """
    refused = _is_refusal(answer)

    if case.should_refuse:
        return not refused

    if refused:
        return False

    context = _context_text(hits).lower()
    answer_no_cite = _strip_citations(answer)
    answer_lower = answer_no_cite.lower()

    for keyword in case.forbidden_answer_keywords:
        if _contains_keyword(answer_lower, keyword) and not _contains_keyword(context, keyword):
            return True

    context_numbers = _number_tokens(context)
    for number in _number_tokens(answer_no_cite):
        if _is_year(number):
            continue
        if number not in context_numbers:
            return True

    return False


def redundancy(hits: list[SearchHit], vectors_by_id: dict) -> float:
    """Max pairwise cosine among final chunks (1.0 == exact duplicate).

    `vectors_by_id` maps chunk id -> L2-normalised embedding, so dot product is
    cosine. Returns 0.0 when fewer than two vectors are available.
    """
    import numpy as np

    vecs = [vectors_by_id[h.chunk.id] for h in hits if h.chunk.id in vectors_by_id]
    if len(vecs) < 2:
        return 0.0
    matrix = np.vstack(vecs)
    sim = matrix @ matrix.T
    np.fill_diagonal(sim, -1.0)
    return float(sim.max())


def source_accuracy(hits: list[SearchHit], case: EvalCase) -> float:
    """Fraction of cases where an expected source filename appears in top-k."""
    if not case.expected_source_filenames:
        return 1.0
    if not hits:
        return 0.0
    filenames = {hit.chunk.filename for hit in hits}
    return (
        1.0
        if any(name in filenames for name in case.expected_source_filenames)
        else 0.0
    )


def score_case(
    case: EvalCase,
    answer: str,
    hits: list[SearchHit],
    redundancy_score: float = 0.0,
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
        source_accuracy=source_accuracy(hits, case),
        redundancy=redundancy_score,
    )
