"""Retrieval boosts for EC2 instance-type specification documents.

Dense embeddings often rank repetitive TOC / marketing passages above spec
tables. When a query names a concrete instance type (e.g. ``m5.xlarge``) or
asks about memory-optimized families, promote chunks that contain the matching
specification text into the final top-k.
"""
from __future__ import annotations

import re

from app.core.vector_store import SearchHit

# e.g. m5.xlarge, r6i.4xlarge, c5n.18xlarge
_INSTANCE_TYPE_RE = re.compile(
    r"\b([a-z]\d+[a-z]?\.[a-z][a-z0-9]*)\b", re.IGNORECASE
)

_MEMORY_FAMILY_QUERY = re.compile(
    r"\bmemory[- ]?(?:intensive|optimized)\b|\b(?:large|high)\s+(?:memory|ram)\b",
    re.IGNORECASE,
)


def extract_instance_types(query: str) -> list[str]:
    """Return deduplicated instance type ids mentioned in the query."""
    seen: set[str] = set()
    out: list[str] = []
    for match in _INSTANCE_TYPE_RE.finditer(query):
        token = match.group(1).lower()
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


def _chunk_contains_instance_type(chunk_text: str, instance_types: list[str]) -> bool:
    lower = chunk_text.lower()
    return any(token in lower for token in instance_types)


def _chunk_is_memory_family_definition(chunk_text: str) -> bool:
    lower = chunk_text.lower()
    return (
        "memory optimized" in lower
        and "designed" in lower
        and ("workload" in lower or "data sets in memory" in lower)
    )


def promote_spec_hits(query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
    """Ensure spec-relevant chunks appear in the final top-k when possible."""
    if not hits:
        return hits

    trimmed = hits[:top_k]
    instance_types = extract_instance_types(query)
    memory_query = _MEMORY_FAMILY_QUERY.search(query) is not None

    if not instance_types and not memory_query:
        return trimmed

    def matches(hit: SearchHit) -> bool:
        if instance_types and _chunk_contains_instance_type(hit.chunk.text, instance_types):
            return True
        return memory_query and _chunk_is_memory_family_definition(hit.chunk.text)

    if any(matches(h) for h in trimmed):
        return trimmed

    for hit in hits:
        if matches(hit):
            promoted = [hit, *[h for h in trimmed if h.chunk.id != hit.chunk.id]]
            return promoted[:top_k]

    return trimmed
