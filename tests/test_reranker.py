"""Tests for cross-encoder reranking."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.core.reranker import Reranker
from app.core.vector_store import SearchHit, StoredChunk


def _hit(text: str, score: float, chunk_index: int = 0) -> SearchHit:
    return SearchHit(
        chunk=StoredChunk(
            id=f"doc:{chunk_index}",
            document_id="doc",
            filename="test.pdf",
            chunk_index=chunk_index,
            page=1,
            text=text,
        ),
        score=score,
    )


@patch("sentence_transformers.CrossEncoder")
def test_rerank_reorders_by_cross_encoder_scores(mock_cross_encoder_cls):
    settings = MagicMock()
    settings.rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    mock_model = MagicMock()
    mock_model.predict.return_value = [0.1, 0.9, 0.4]
    mock_cross_encoder_cls.return_value = mock_model

    reranker = Reranker(settings)
    hits = [
        _hit("pro-rated annual leave rules", 0.58, 0),
        _hit("annual leave entitlement 14 days", 0.49, 1),
        _hit("medical leave policy", 0.45, 2),
    ]

    reranked = reranker.rerank("How many annual leave days?", hits)

    assert [h.chunk.chunk_index for h in reranked] == [1, 2, 0]
    assert reranked[0].score == 0.9
