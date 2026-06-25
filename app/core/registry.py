"""Document registry.

Lightweight JSON-backed catalogue of ingested documents so the API can list
and delete them. The authoritative chunk data lives in the vector store; this
just tracks document-level metadata.
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class DocumentRecord:
    id: str
    filename: str
    content_type: str
    num_chunks: int
    num_chars: int
    uploaded_at: str


class DocumentRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._records: dict[str, DocumentRecord] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._records = {r["id"]: DocumentRecord(**r) for r in raw}

    def _persist(self) -> None:
        self.path.write_text(
            json.dumps([asdict(r) for r in self._records.values()], ensure_ascii=False),
            encoding="utf-8",
        )

    def add(self, record: DocumentRecord) -> None:
        with self._lock:
            self._records[record.id] = record
            self._persist()

    def remove(self, document_id: str) -> bool:
        with self._lock:
            if document_id in self._records:
                del self._records[document_id]
                self._persist()
                return True
            return False

    def get(self, document_id: str) -> DocumentRecord | None:
        return self._records.get(document_id)

    def list(self) -> list[DocumentRecord]:
        return sorted(self._records.values(), key=lambda r: r.uploaded_at, reverse=True)

    @property
    def count(self) -> int:
        return len(self._records)
