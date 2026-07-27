"""Tests for BM25 hybrid retrieval helpers."""
from __future__ import annotations

from app.core.bm25_index import (
    BM25Index,
    demote_overview_hits,
    overview_penalty,
    reciprocal_rank_fusion,
    tokenize,
)
from app.core.vector_store import SearchHit, StoredChunk


def _chunk(cid: str, text: str, document_id: str = "doc") -> StoredChunk:
    return StoredChunk(
        id=cid,
        document_id=document_id,
        filename=f"{document_id}.pdf",
        chunk_index=int(cid.split(":")[-1]) if ":" in cid else 0,
        page=1,
        text=text,
    )


def _hit(cid: str, text: str, score: float) -> SearchHit:
    return SearchHit(chunk=_chunk(cid, text), score=score)


def test_tokenize_keeps_code_like_tokens():
    tokens = tokenize("Use OAuth2PasswordBearer and create_access_token()")
    assert "oauth2passwordbearer" in tokens
    assert "create_access_token" in tokens


def test_bm25_ranks_exact_api_symbol_above_overview():
    overview = _chunk(
        "d:0",
        "By the end of this chapter, you will learn about OAuth2 and JWT authentication.",
    )
    howto = _chunk(
        "d:1",
        "from fastapi.security import OAuth2PasswordBearer\n"
        "oauth2_scheme = OAuth2PasswordBearer(tokenUrl='token')\n"
        "def create_access_token(data: dict): ...",
    )
    idx = BM25Index()
    idx.rebuild([overview, howto])
    hits = idx.search("OAuth2PasswordBearer create_access_token JWT", top_k=2)
    assert hits
    assert hits[0].chunk.id == "d:1"


def test_rrf_merges_dense_and_bm25_rankings():
    # Dense prefers overview; BM25 strongly prefers the code chunk (top-1 twice).
    dense = [
        _hit("a", "overview oauth jwt", 0.9),
        _hit("b", "code OAuth2PasswordBearer", 0.5),
        _hit("c", "other", 0.4),
    ]
    sparse = [
        _hit("b", "code OAuth2PasswordBearer", 12.0),
        _hit("c", "other", 4.0),
        _hit("a", "overview oauth jwt", 3.0),
    ]
    fused = reciprocal_rank_fusion([dense, sparse], rrf_k=60)
    assert fused[0].chunk.id == "b"


def test_overview_penalty_detects_chapter_intro():
    intro = "By the end of this chapter, you will be able to secure FastAPI with OAuth2."
    code = "oauth2_scheme = OAuth2PasswordBearer(tokenUrl='token')"
    assert overview_penalty(intro) > overview_penalty(code)


def test_demote_overview_hits_reorders():
    hits = [
        _hit("intro", "By the end of this chapter, we will cover JWT.", 10.0),
        _hit("code", "OAuth2PasswordBearer tokenUrl create_access_token", 9.0),
    ]
    demoted = demote_overview_hits(hits, strength=0.7)
    assert demoted[0].chunk.id == "code"
