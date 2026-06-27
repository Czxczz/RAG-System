"""Unit tests for query rewriting, MMR diversity, and context grouping."""
from __future__ import annotations

import numpy as np

from app.config import Settings
from app.core.context_grouping import (
    build_flat_context,
    build_grouped_context,
    group_hits,
)
from app.core.diversity import dedupe_by_text, mmr_order, mmr_rerank
from app.core.query_rewriter import QueryRewriter
from app.core.vector_store import SearchHit, StoredChunk


def _hit(doc: str, chunk_index: int, text: str, score: float = 0.8, page: int = 1) -> SearchHit:
    return SearchHit(
        chunk=StoredChunk(
            id=f"{doc}:{chunk_index}",
            document_id=doc,
            filename=f"{doc}.pdf",
            chunk_index=chunk_index,
            page=page,
            text=text,
        ),
        score=score,
    )


# ── Query rewriting (heuristic path, no LLM) ─────────────────
class _NoLLM:
    def has_llm(self, mode: str = "auto") -> bool:
        return False


def test_rewriter_disabled_returns_original_only():
    settings = Settings(query_rewrite_enabled=False)
    rw = QueryRewriter(settings, _NoLLM())
    assert rw.rewrite("How do I reset my password?") == ["How do I reset my password?"]


def test_rewriter_heuristic_produces_unique_variants():
    settings = Settings(query_rewrite_use_llm=False, query_rewrite_num_variants=3)
    rw = QueryRewriter(settings, _NoLLM())
    variants = rw.rewrite("How do I reset my password?")
    assert variants[0] == "How do I reset my password?"  # original kept first
    assert len(variants) == len(set(v.lower() for v in variants))  # deduped
    assert len(variants) <= 4  # original + up to 3 variants
    # A keyword-only variant should drop question words.
    assert any("reset" in v and "password" in v and "how" not in v.lower() for v in variants)


# ── MMR diversity ────────────────────────────────────────────
def test_mmr_order_prefers_diverse_second_pick():
    # Items 0 and 1 are near-identical; item 2 is distinct but less relevant.
    relevance = [1.0, 0.95, 0.6]
    similarity = np.array(
        [
            [1.0, 0.99, 0.1],
            [0.99, 1.0, 0.1],
            [0.1, 0.1, 1.0],
        ]
    )
    order = mmr_order(relevance, similarity, lambda_=0.5, k=2)
    assert order[0] == 0  # most relevant first
    assert order[1] == 2  # diversity beats the near-duplicate item 1


def test_mmr_rerank_drops_redundant_chunk():
    hits = [
        _hit("docA", 0, "The sky is blue.", score=1.0),
        _hit("docA", 1, "The sky is blue.", score=0.98),  # duplicate-ish
        _hit("docB", 0, "Mars is red.", score=0.7),
    ]
    vectors = {
        "docA:0": np.array([1.0, 0.0, 0.0], dtype="float32"),
        "docA:1": np.array([0.99, 0.01, 0.0], dtype="float32"),
        "docB:0": np.array([0.0, 1.0, 0.0], dtype="float32"),
    }
    out = mmr_rerank(hits, vectors, lambda_=0.5, k=2)
    ids = {h.chunk.id for h in out}
    assert "docA:0" in ids and "docB:0" in ids  # diverse pair chosen
    assert "docA:1" not in ids


def test_mmr_rerank_falls_back_when_vectors_missing():
    hits = [_hit("docA", 0, "x", 1.0), _hit("docB", 0, "y", 0.5), _hit("docC", 0, "z", 0.4)]
    out = mmr_rerank(hits, {}, lambda_=0.7, k=2)
    assert [h.chunk.id for h in out] == ["docA:0", "docB:0"]  # identity top-k


def test_mmr_dedup_threshold_drops_near_duplicates():
    # Items 0 and 1 are near-identical (cosine ~0.99); a hard 0.95 cap must
    # drop item 1 even though it is highly relevant.
    relevance = [1.0, 0.99, 0.5]
    similarity = np.array(
        [
            [1.0, 0.99, 0.2],
            [0.99, 1.0, 0.2],
            [0.2, 0.2, 1.0],
        ]
    )
    order = mmr_order(relevance, similarity, lambda_=0.7, k=3, dedup_threshold=0.95)
    assert order[0] == 0
    assert 1 not in order  # near-duplicate hard-dropped
    assert order[1] == 2


def test_mmr_rerank_dedup_can_return_fewer_than_k():
    # All three chunks are mutual exact duplicates -> only one should survive.
    hits = [_hit("docA", 0, "dup", 1.0), _hit("docA", 1, "dup", 0.9), _hit("docA", 2, "dup", 0.8)]
    vectors = {
        "docA:0": np.array([1.0, 0.0], dtype="float32"),
        "docA:1": np.array([1.0, 0.0], dtype="float32"),
        "docA:2": np.array([1.0, 0.0], dtype="float32"),
    }
    out = mmr_rerank(hits, vectors, lambda_=0.6, k=3, dedup_threshold=0.88)
    assert len(out) == 1
    assert out[0].chunk.id == "docA:0"


def test_dedupe_by_text_drops_boilerplate():
    hits = [
        _hit("docA", 0, "Amazon EC2 User Guide", score=1.0),
        _hit("docA", 1, "Amazon EC2 User Guide", score=0.95),
        _hit("docA", 2, "Instance types include t3.micro and m5.large.", score=0.8),
    ]
    out = dedupe_by_text(hits, jaccard_threshold=0.85)
    assert len(out) == 2
    assert out[0].chunk.chunk_index == 0
    assert out[1].chunk.chunk_index == 2


def test_dedupe_by_text_drops_high_jaccard_overlap():
    hits = [
        _hit("docA", 0, "Enable IMDSv2 on your instance metadata service.", score=1.0),
        _hit(
            "docA",
            1,
            "Enable IMDSv2 on your instance metadata service now.",
            score=0.9,
        ),
        _hit("docA", 2, "Security groups control inbound traffic.", score=0.7),
    ]
    out = dedupe_by_text(hits, jaccard_threshold=0.85)
    assert len(out) == 2
    assert out[1].chunk.chunk_index == 2


# ── Context grouping ─────────────────────────────────────────
def test_group_hits_orders_by_reading_order_within_document():
    hits = [
        _hit("docA", 2, "third"),
        _hit("docB", 0, "other"),
        _hit("docA", 0, "first"),
    ]
    groups = group_hits(hits)
    # docA appears first (its best hit ranked first), then docB.
    assert [g.document_id for g in groups] == ["docA", "docB"]
    # Within docA, chunks are sorted by chunk_index.
    assert [h.chunk.chunk_index for h in groups[0].hits] == [0, 2]


def test_build_grouped_context_keeps_marker_alignment():
    hits = [
        _hit("docA", 0, "Alpha statement."),
        _hit("docB", 0, "Bravo statement."),
        _hit("docA", 1, "Alpha continued."),
    ]
    context, ordered = build_grouped_context(hits)
    # Markers are assigned in presentation order; ordered_hits mirrors them.
    assert "[1]" in context and "[2]" in context and "[3]" in context
    assert len(ordered) == 3
    # docA's two chunks are grouped and contiguous (markers 1 and 2).
    assert ordered[0].chunk.document_id == "docA"
    assert ordered[1].chunk.document_id == "docA"
    assert ordered[2].chunk.document_id == "docB"
    assert "### Source: docA.pdf" in context


def test_build_grouped_context_dedupes_adjacent_overlap():
    # Adjacent chunks share the overlap sentence "Shared overlap line."
    hits = [
        _hit("docA", 0, "First unique part. Shared overlap line."),
        _hit("docA", 1, "Shared overlap line. Second unique part."),
    ]
    context, _ = build_grouped_context(hits)
    # The shared sentence should appear only once after de-duplication.
    assert context.count("Shared overlap line.") == 1
    assert "Second unique part." in context


def test_build_grouped_context_stubs_exact_duplicate():
    hits = [
        _hit("docA", 0, "Same body text here."),
        _hit("docA", 1, "Same body text here."),
    ]
    context, ordered = build_grouped_context(hits)
    assert len(ordered) == 2  # citation markers preserved
    assert context.count("Same body text here.") == 1
    assert "See preceding passage [1]" in context


def test_build_flat_context_is_relevance_order():
    hits = [_hit("docB", 0, "b"), _hit("docA", 0, "a")]
    context, ordered = build_flat_context(hits)
    assert [h.chunk.document_id for h in ordered] == ["docB", "docA"]
    assert context.index("[1]") < context.index("[2]")
