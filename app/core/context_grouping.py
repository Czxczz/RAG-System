"""Context grouping for coherent, citation-stable prompts.

Retrieved chunks arrive in relevance order, which often interleaves snippets
from different documents and presents same-document chunks out of reading
order. That hurts coherence: the LLM jumps between unrelated sources.

This module regroups the selected chunks so that:
  * chunks from the same document are presented together, under one header;
  * within a document they appear in natural reading order (page, chunk index);
  * documents are ordered by their most relevant chunk (relevance is preserved);
  * overlapping text shared by adjacent chunks is de-duplicated.

Each chunk keeps its own ``[n]`` citation marker, assigned in presentation
order, so the marker -> chunk mapping (used for citations and evaluation)
stays exact.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.vector_store import SearchHit


@dataclass
class ContextGroup:
    document_id: str
    filename: str
    hits: list[SearchHit]  # ordered by (page, chunk_index)


def group_hits(hits: list[SearchHit]) -> list[ContextGroup]:
    """Group hits by source document, preserving relevance priority."""
    order: list[str] = []
    buckets: dict[str, list[SearchHit]] = {}
    for hit in hits:
        doc_id = hit.chunk.document_id
        if doc_id not in buckets:
            buckets[doc_id] = []
            order.append(doc_id)
        buckets[doc_id].append(hit)

    groups: list[ContextGroup] = []
    for doc_id in order:
        doc_hits = sorted(
            buckets[doc_id],
            key=lambda h: (h.chunk.page or 0, h.chunk.chunk_index),
        )
        groups.append(
            ContextGroup(
                document_id=doc_id,
                filename=doc_hits[0].chunk.filename,
                hits=doc_hits,
            )
        )
    return groups


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _dedupe_overlap(previous: SearchHit, current: SearchHit) -> str:
    """Drop leading sentences of `current` that repeat the tail of `previous`.

    Only applied to adjacent chunks (consecutive index, same document), which
    is exactly where the ingestion overlap window creates duplicate text.
    """
    if (
        previous.chunk.document_id != current.chunk.document_id
        or current.chunk.chunk_index != previous.chunk.chunk_index + 1
    ):
        return current.chunk.text

    prev_sentences = _split_sentences(previous.chunk.text)
    curr_sentences = _split_sentences(current.chunk.text)
    if not prev_sentences or not curr_sentences:
        return current.chunk.text

    prev_tail = set(prev_sentences[-5:])
    start = 0
    while start < len(curr_sentences) and curr_sentences[start] in prev_tail:
        start += 1

    trimmed = " ".join(curr_sentences[start:]).strip()
    return trimmed or current.chunk.text


def build_grouped_context(
    hits: list[SearchHit],
) -> tuple[str, list[SearchHit]]:
    """Build the grouped CONTEXT block and the marker-aligned hit ordering.

    Returns ``(context_text, ordered_hits)`` where ``ordered_hits[i]`` is the
    chunk referenced by marker ``[i + 1]``.
    """
    groups = group_hits(hits)
    blocks: list[str] = []
    ordered: list[SearchHit] = []
    marker = 1

    for group in groups:
        lines = [f"### Source: {group.filename}"]
        prev: SearchHit | None = None
        for hit in group.hits:
            text = _dedupe_overlap(prev, hit) if prev else hit.chunk.text
            loc = f"p.{hit.chunk.page}" if hit.chunk.page is not None else f"chunk {hit.chunk.chunk_index}"
            lines.append(f"[{marker}] ({loc})\n{text}")
            ordered.append(hit)
            marker += 1
            prev = hit
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks), ordered


def build_flat_context(hits: list[SearchHit]) -> tuple[str, list[SearchHit]]:
    """Ungrouped fallback: one numbered block per hit, in relevance order."""
    blocks: list[str] = []
    for i, hit in enumerate(hits, start=1):
        loc = hit.chunk.filename
        if hit.chunk.page is not None:
            loc += f", p.{hit.chunk.page}"
        blocks.append(f"[{i}] (source: {loc})\n{hit.chunk.text}")
    return "\n\n".join(blocks), list(hits)
