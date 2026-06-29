"""Unit tests for the post-generation answer validation gate."""
from __future__ import annotations

from types import SimpleNamespace

from app.config import Settings
from app.core.answer_validation import validate_answer
from app.core.orchestrator import RAGOrchestrator
from app.core.vector_store import SearchHit, StoredChunk


def _hit(text: str, chunk_index: int = 0) -> SearchHit:
    return SearchHit(
        chunk=StoredChunk(
            id=f"doc:{chunk_index}",
            document_id="doc",
            filename="ec2-ug.pdf",
            chunk_index=chunk_index,
            page=1,
            text=text,
        ),
        score=0.8,
    )


def test_supported_citation_passes():
    hits = [_hit("An Elastic IP address is a static public IPv4 address.", 0)]
    answer = "An Elastic IP address is static and public [1]."
    result = validate_answer(answer, hits, min_support=0.5)
    assert result.passed is True
    assert result.support == 1.0
    assert result.notes == []


def test_out_of_range_marker_fails():
    hits = [_hit("Security groups control inbound traffic.", 0)]
    answer = "Configure inbound rules [1]. See also the appendix [5]."
    result = validate_answer(answer, hits, min_support=0.5)
    assert result.passed is False
    assert 5 in result.invalid_markers


def test_unsupported_citation_lowers_support():
    hits = [
        _hit("Spot Instances can be interrupted by EC2.", 0),
        _hit("Completely unrelated text about billing tax codes.", 1),
    ]
    # [2] cites a chunk that doesn't support the claim about hibernation.
    answer = "Spot Instances are interrupted [1]. They always hibernate first [2]."
    result = validate_answer(answer, hits, min_support=0.75)
    assert result.support < 0.75
    assert result.passed is False


def test_answer_without_citations_but_with_claims_fails():
    hits = [_hit("Some relevant context about instances.", 0)]
    answer = (
        "Instances can be launched in any region and configured with many "
        "different options depending on your workload requirements."
    )
    result = validate_answer(answer, hits, min_support=0.5)
    assert result.passed is False
    assert result.notes


def test_short_answer_without_citation_is_lenient():
    hits = [_hit("context", 0)]
    answer = "Yes."
    result = validate_answer(answer, hits, min_support=0.5)
    assert result.passed is True


# ── Retrieval confidence gate ────────────────────────────────
def _gate(hits, *, enabled=True, min_score=0.0) -> bool:
    fake = SimpleNamespace(
        settings=Settings(
            retrieval_gate_enabled=enabled,
            retrieval_gate_min_score=min_score,
        )
    )
    return RAGOrchestrator._retrieval_below_gate(fake, hits)


def test_retrieval_gate_blocks_low_score():
    hits = [_hit("weakly related", 0)]
    hits[0] = SearchHit(chunk=hits[0].chunk, score=-2.0)
    assert _gate(hits, min_score=0.0) is True


def test_retrieval_gate_allows_confident_hit():
    hits = [_hit("strongly relevant", 0)]
    hits[0] = SearchHit(chunk=hits[0].chunk, score=7.5)
    assert _gate(hits, min_score=0.0) is False


def test_retrieval_gate_disabled_never_blocks():
    hits = [_hit("weak", 0)]
    hits[0] = SearchHit(chunk=hits[0].chunk, score=-5.0)
    assert _gate(hits, enabled=False, min_score=0.0) is False
