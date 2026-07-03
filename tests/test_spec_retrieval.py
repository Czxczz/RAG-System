"""Tests for instance-type spec retrieval boosts."""
from __future__ import annotations

from app.core.spec_retrieval import extract_instance_types, promote_spec_hits
from app.core.vector_store import SearchHit, StoredChunk


def _hit(text: str, chunk_id: str = "a:0", score: float = 0.5) -> SearchHit:
    return SearchHit(
        chunk=StoredChunk(
            id=chunk_id,
            document_id="a",
            filename="ec2-types.pdf",
            chunk_index=0,
            page=1,
            text=text,
        ),
        score=score,
    )


def test_extract_instance_types():
    q = "How many vCPUs does the m5.xlarge instance type provide?"
    assert extract_instance_types(q) == ["m5.xlarge"]


def test_promote_spec_hits_injects_instance_type_chunk():
    toc = _hit("General Purpose M5 M5d M6g instance families overview", "a:0", 0.9)
    spec = _hit(
        "m5.xlarge 16.00 GiB Intel Xeon 4 vCPUs performance specifications",
        "a:1",
        0.4,
    )
    hits = [toc, spec]
    promoted = promote_spec_hits(
        "How many vCPUs does the m5.xlarge instance type provide?",
        hits,
        top_k=1,
    )
    assert promoted[0].chunk.id == "a:1"


def test_promote_spec_hits_injects_memory_family_definition():
    toc = _hit("Memory Optimized: R5 | R5a | R5ad family list", "a:0", 0.9)
    definition = _hit(
        "Memory optimized – Designed to deliver fast performance for workloads "
        "that process large data sets in memory.",
        "a:1",
        0.4,
    )
    promoted = promote_spec_hits(
        "Which EC2 instance family is designed for memory-intensive workloads?",
        [toc, definition],
        top_k=1,
    )
    assert promoted[0].chunk.id == "a:1"
