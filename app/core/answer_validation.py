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

_STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "on", "for", "is", "are",
    "was", "were", "be", "with", "at", "by", "from", "that", "this", "it", "you",
    "your", "can", "will", "if", "as", "when", "which",
}

DISCLAIMER = (
    "\n\n_Note: parts of this answer may not be fully supported by the cited "
    "sources; please verify against the referenced passages._"
)


# Matches [1], [2, 4], [1, 2, 3] — Gemini often emits grouped markers.
_CITATION_GROUP_RE = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")


@dataclass
class ValidationResult:
    passed: bool
    support: float  # fraction of citation markers backed by their cited chunk
    invalid_markers: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _citation_groups(answer: str) -> list[tuple[str, list[int]]]:
    """Return (claim line, markers) for each inline citation group."""
    pattern = re.compile(
        r"(.+?)\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]", re.IGNORECASE | re.DOTALL
    )
    groups: list[tuple[str, list[int]]] = []
    for match in pattern.finditer(answer):
        claim = match.group(1).split("\n")[-1]
        markers = [int(part.strip()) for part in match.group(2).split(",")]
        groups.append((claim, markers))
    return groups


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
    groups = _citation_groups(answer)
    if not groups:
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
    for claim, markers in groups:
        out_of_range = [m for m in markers if m < 1 or m > len(hits)]
        invalid_markers.extend(out_of_range)
        valid = [m for m in markers if 1 <= m <= len(hits)]
        if not valid:
            continue
        tokens = _claim_tokens(claim)
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

    support = supported / len(groups)
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
