"""Query rewriting for multi-query retrieval.

A single phrasing of a question often misses relevant chunks that use different
vocabulary. We expand the query into several variants, retrieve for each, and
fuse the candidate pools downstream. This lifts recall (and, after reranking,
precision) without changing the index.

Two strategies, chosen automatically:
  * LLM      -> ask the configured LLM for paraphrases / sub-questions.
  * heuristic-> deterministic, offline fallback (keyword + question-word
                stripping). Always available, no network, no cost.

The original query is always kept as the first variant.
"""
from __future__ import annotations

import re

from app.config import Settings
from app.core.llm_router import LLMRouter

_QUESTION_WORDS = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "is", "are", "was", "were", "do", "does", "did", "can", "could", "should",
    "would", "will", "the", "a", "an", "of", "to", "in", "on", "for", "and",
    "or", "me", "tell", "about", "please",
}

_REWRITE_SYSTEM = (
    "You rewrite a user's question into alternative search queries for a "
    "document retrieval system. Produce diverse paraphrases and focused "
    "sub-questions that use synonyms and related terminology. Output ONLY the "
    "queries, one per line, with no numbering, quotes, or commentary."
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _keyword_variant(query: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", query.lower())
    kept = [t for t in tokens if t not in _QUESTION_WORDS and len(t) > 2]
    return " ".join(kept)


# Broad taxonomy questions ("categories of instance types") often retrieve
# overview chunks that point to an external guide instead of the list section.
# A deterministic variant using the family names lifts the right passages.
_TAXONOMY_QUERY = re.compile(
    r"\b(?:categor(?:y|ies)|families|types?|kinds?)\b.*\b(?:instance|ec2)\b"
    r"|\b(?:instance|ec2)\b.*\b(?:categor(?:y|ies)|families|types?|kinds?)\b",
    re.IGNORECASE,
)
_TAXONOMY_VARIANT = (
    "EC2 instance type families general purpose compute optimized "
    "memory optimized storage optimized accelerated computing"
)

_MEMORY_FAMILY_QUERY = re.compile(
    r"\bmemory[- ]?(?:intensive|optimized)\b|\b(?:large|high)\s+(?:memory|ram)\b",
    re.IGNORECASE,
)
_MEMORY_FAMILY_VARIANT = (
    "memory optimized designed deliver fast performance workloads "
    "process large data sets in memory"
)

_INSTANCE_SPEC_QUERY = re.compile(
    r"\b(vcpu|vcpus|memory|gi[b]|ram|bandwidth|network|storage|iops|"
    r"processor|cores?|threads?)\b",
    re.IGNORECASE,
)
_INSTANCE_TYPE_IN_QUERY = re.compile(
    r"\b[a-z]\d+[a-z]?\.[a-z][a-z0-9]*\b", re.IGNORECASE
)


def _taxonomy_variant(query: str) -> str | None:
    if _TAXONOMY_QUERY.search(query):
        return _TAXONOMY_VARIANT
    return None


def _memory_family_variant(query: str) -> str | None:
    if _MEMORY_FAMILY_QUERY.search(query):
        return _MEMORY_FAMILY_VARIANT
    return None


def _instance_spec_variant(query: str) -> str | None:
    if not _INSTANCE_TYPE_IN_QUERY.search(query):
        return None
    if not _INSTANCE_SPEC_QUERY.search(query):
        return None
    types = _INSTANCE_TYPE_IN_QUERY.findall(query)
    if not types:
        return None
    return f"{types[0]} performance specifications vCPUs memory processor"


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        norm = item.lower()
        if item and norm not in seen:
            seen.add(norm)
            out.append(item)
    return out


class QueryRewriter:
    def __init__(self, settings: Settings, llm: LLMRouter) -> None:
        self.settings = settings
        self.llm = llm

    def rewrite(self, query: str) -> list[str]:
        """Return [original, *variants]; never raises, always >= 1 item."""
        query = _normalize(query)
        if not self.settings.query_rewrite_enabled:
            return [query]

        n = max(0, self.settings.query_rewrite_num_variants)
        variants: list[str] = []

        if self.settings.query_rewrite_use_llm and self.llm.has_llm("auto"):
            try:
                variants = self._llm_variants(query, n)
            except Exception:
                variants = []  # fall back silently to heuristics

        if not variants:
            variants = self._heuristic_variants(query)

        taxonomy = _taxonomy_variant(query)
        if taxonomy:
            variants.insert(0, taxonomy)

        memory_family = _memory_family_variant(query)
        if memory_family:
            variants.insert(0, memory_family)

        instance_spec = _instance_spec_variant(query)
        if instance_spec:
            variants.insert(0, instance_spec)

        # Allow extra slots when deterministic variants are injected so they are
        # not truncated by the usual variant budget.
        extra = sum(1 for v in (taxonomy, memory_family, instance_spec) if v)
        limit = n + 1 + extra
        return _dedupe_preserve_order([query, *variants])[:limit]

    # ── Strategies ───────────────────────────────────────────
    def _llm_variants(self, query: str, n: int) -> list[str]:
        if n <= 0:
            return []
        user_prompt = (
            f"Question: {query}\n\n"
            f"Write {n} alternative search queries (one per line)."
        )
        text, _ = self.llm.generate(_REWRITE_SYSTEM, user_prompt, mode="auto")
        lines = [
            re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", line).strip().strip('"')
            for line in text.splitlines()
        ]
        return [_normalize(line) for line in lines if line.strip()]

    def _heuristic_variants(self, query: str) -> list[str]:
        variants: list[str] = []
        keywords = _keyword_variant(query)
        if keywords and keywords != query.lower():
            variants.append(keywords)
        # Strip leading question word ("How do I reset?" -> "reset")
        stripped = re.sub(
            r"^\s*(what|which|who|when|where|why|how|is|are|do|does|did|can|"
            r"could|should|would|will)\b[\s,]*",
            "",
            query,
            flags=re.IGNORECASE,
        ).strip(" ?.")
        if stripped and stripped.lower() != query.lower():
            variants.append(stripped)
        return variants
