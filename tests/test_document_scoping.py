"""Tests for per-document retrieval scoping."""
from __future__ import annotations

import numpy as np
import pytest

from app.core.vector_store import SearchHit, StoredChunk, VectorStore


def _chunk(doc_id: str, filename: str, text: str, index: int = 0) -> StoredChunk:
    return StoredChunk(
        id=f"{doc_id}:{index}",
        document_id=doc_id,
        filename=filename,
        chunk_index=index,
        page=1,
        text=text,
    )


def test_vector_store_search_filters_by_document_ids(tmp_path):
    store = VectorStore(
        dimension=4,
        index_path=tmp_path / "index.faiss",
        metadata_path=tmp_path / "meta.json",
    )
    chunks = [
        _chunk("doc-a", "a.pdf", "placement group cluster spread"),
        _chunk("doc-b", "b.pdf", "m5.xlarge vCPU memory optimized"),
    ]
    vectors = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype="float32",
    )
    store.add(vectors, chunks)

    all_hits = store.search(np.array([1.0, 0.0, 0.0, 0.0]), top_k=2)
    assert len(all_hits) == 2

    scoped = store.search(
        np.array([1.0, 0.0, 0.0, 0.0]),
        top_k=2,
        document_ids={"doc-b"},
    )
    assert len(scoped) == 1
    assert scoped[0].chunk.document_id == "doc-b"
    assert scoped[0].chunk.filename == "b.pdf"


def test_normalize_document_scope_rejects_unknown_ids(tmp_path):
    from app.config import Settings
    from app.core.orchestrator import RAGOrchestrator
    from app.core.registry import DocumentRegistry
    from app.core.vector_store import VectorStore

    settings = Settings(data_dir=tmp_path, query_rewrite_enabled=False)
    settings.ensure_dirs()
    store = VectorStore(
        dimension=4,
        index_path=settings.index_path,
        metadata_path=settings.metadata_path,
    )
    registry = DocumentRegistry(settings.documents_path)

    class _FakeEmb:
        pass

    orch = RAGOrchestrator(
        settings=settings,
        embeddings=_FakeEmb(),
        store=store,
        registry=registry,
        reranker=None,
    )

    assert orch.normalize_document_scope(None) is None
    with pytest.raises(ValueError, match="Unknown document_ids"):
        orch.normalize_document_scope(["missing-doc"])
