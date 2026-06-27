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

        return _dedupe_preserve_order([query, *variants])[: n + 1]

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
