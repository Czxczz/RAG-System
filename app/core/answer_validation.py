"""Post-generation answer validation gate.

The LLM is instructed to ground every claim in the retrieved context with inline
``[n]`` citations, but it can still drift: cite a passage number that does not
exist, or assert something the cited chunk does not support. This module checks
the generated answer against the retrieved chunks and reports whether the answer
is adequately grounded, without rewriting the answer's substance.

It is deliberately lightweight (lexical overlap, no extra model call) so it adds
negligible latency and never blocks a good answer — at worst it appends a short
disclaimer and flags the answer as not fully grounded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.vector_store import SearchHit

_CITATION_RE = re.compile(r"\[(\d+)\]")
_STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "on", "for", "is", "are",
    "was", "were", "be", "with", "at", "by", "from", "that", "this", "it", "you",
    "your", "can", "will", "if", "as", "when", "which",
}

DISCLAIMER = (
    "\n\n_Note: parts of this answer may not be fully supported by the cited "
    "sources; please verify against the referenced passages._"
)


@dataclass
class ValidationResult:
    passed: bool
    support: float  # fraction of citation markers backed by their cited chunk
    invalid_markers: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _claim_before_marker(answer: str, marker: int) -> str:
    pattern = re.compile(rf"(.+?)\[{marker}\]", re.IGNORECASE | re.DOTALL)
    match = pattern.search(answer)
    if not match:
        return ""
    return match.group(1).split("\n")[-1]


def _claim_tokens(claim: str) -> set[str]:
    return {
        word.lower()
        for word in re.findall(r"[A-Za-z0-9]+", claim)
        if word.lower() not in _STOPWORDS and len(word) > 2
    }


def validate_answer(
    answer: str,
    hits: list[SearchHit],
    *,
    min_support: float = 0.5,
) -> ValidationResult:
    """Check that an answer's citations exist and are lexically supported.

    Returns a :class:`ValidationResult`. An answer with no citation markers and
    no hits is treated as trivially valid (e.g. a refusal handled upstream).
    """
    markers = [int(m) for m in _CITATION_RE.findall(answer)]
    if not markers:
        # No citations to verify. Only flag when the answer makes claims but
        # cites nothing despite context being available.
        if hits and len(answer.split()) > 12:
            return ValidationResult(
                passed=False,
                support=0.0,
                notes=["Answer makes claims but contains no [n] citations."],
            )
        return ValidationResult(passed=True, support=1.0)

    invalid_markers: list[int] = []
    supported = 0
    for marker in markers:
        if marker < 1 or marker > len(hits):
            invalid_markers.append(marker)
            continue
        chunk_text = hits[marker - 1].chunk.text.lower()
        tokens = _claim_tokens(_claim_before_marker(answer, marker))
        if not tokens:
            supported += 1
            continue
        overlap = sum(1 for token in tokens if token in chunk_text)
        if overlap / len(tokens) >= 0.25:
            supported += 1

    support = supported / len(markers)
    notes: list[str] = []
    if invalid_markers:
        notes.append(
            f"Answer cites passages that do not exist: "
            f"{sorted(set(invalid_markers))}."
        )
    if support < min_support:
        notes.append(
            f"Only {support:.0%} of citations are supported by their sources."
        )

    passed = not invalid_markers and support >= min_support
    return ValidationResult(
        passed=passed,
        support=support,
        invalid_markers=invalid_markers,
        notes=notes,
    )
