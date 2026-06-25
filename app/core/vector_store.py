"""FAISS-backed vector store with on-disk persistence.

Stores normalised embedding vectors in a FAISS inner-product index (cosine
similarity) alongside a parallel JSON metadata list. The index and metadata
are persisted to `DATA_DIR` and reloaded on startup.

Thread-safety: a single process-wide lock guards mutations so concurrent
FastAPI requests don't corrupt the index.
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np


@dataclass
class StoredChunk:
    """Metadata stored alongside each vector (mirrors the FAISS row order)."""

    id: str  # globally unique: f"{document_id}:{chunk_index}"
    document_id: str
    filename: str
    chunk_index: int
    page: int | None
    text: str


@dataclass
class SearchHit:
    chunk: StoredChunk
    score: float


class VectorStore:
    def __init__(self, dimension: int, index_path: Path, metadata_path: Path) -> None:
        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = metadata_path
        self._lock = threading.Lock()
        self._index: faiss.Index
        self._metadata: list[StoredChunk] = []
        self._load_or_init()

    # ── Persistence ──────────────────────────────────────────
    def _load_or_init(self) -> None:
        if self.index_path.exists() and self.metadata_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            raw = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self._metadata = [StoredChunk(**item) for item in raw]
            # Guard against dimension mismatch (e.g. switching embedding model).
            if self._index.d != self.dimension:
                self._reset()
        else:
            self._reset()

    def _reset(self) -> None:
        self._index = faiss.IndexFlatIP(self.dimension)
        self._metadata = []

    def _persist(self) -> None:
        faiss.write_index(self._index, str(self.index_path))
        self.metadata_path.write_text(
            json.dumps([asdict(c) for c in self._metadata], ensure_ascii=False),
            encoding="utf-8",
        )

    # ── Mutations ────────────────────────────────────────────
    def add(self, vectors: np.ndarray, chunks: list[StoredChunk]) -> None:
        if len(chunks) == 0:
            return
        if vectors.shape[0] != len(chunks):
            raise ValueError("vectors and chunks length mismatch")
        with self._lock:
            self._index.add(vectors.astype("float32"))
            self._metadata.extend(chunks)
            self._persist()

    def delete_document(self, document_id: str) -> int:
        """Remove all chunks for a document and rebuild the index."""
        with self._lock:
            keep = [c for c in self._metadata if c.document_id != document_id]
            removed = len(self._metadata) - len(keep)
            if removed == 0:
                return 0
            # IndexFlatIP has no stable selective remove, so rebuild from kept
            # vectors. We reconstruct retained rows before resetting.
            kept_vectors = self._reconstruct(keep)
            self._reset()
            if kept_vectors is not None and kept_vectors.shape[0] > 0:
                self._index.add(kept_vectors)
            self._metadata = keep
            self._persist()
            return removed

    def _reconstruct(self, keep: list[StoredChunk]) -> np.ndarray | None:
        if not keep:
            return None
        # Map kept chunks back to their original row positions.
        id_to_row = {c.id: row for row, c in enumerate(self._metadata)}
        rows = [id_to_row[c.id] for c in keep]
        vectors = np.vstack([self._index.reconstruct(r) for r in rows])
        return vectors.astype("float32")

    # ── Query ────────────────────────────────────────────────
    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchHit]:
        if self._index.ntotal == 0:
            return []
        q = query_vector.reshape(1, -1).astype("float32")
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q, k)
        hits: list[SearchHit] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            hits.append(SearchHit(chunk=self._metadata[idx], score=float(score)))
        return hits

    # ── Introspection ────────────────────────────────────────
    @property
    def num_chunks(self) -> int:
        return self._index.ntotal

    def document_ids(self) -> set[str]:
        return {c.document_id for c in self._metadata}
