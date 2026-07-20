"""Upload API error-handling tests (size, format, corrupt files)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.core.ingestion import CorruptedDocumentError, EmptyDocumentError
from app.core.orchestrator import IngestResult
from app.core.registry import DocumentRecord
from app.dependencies import get_orchestrator
from app.main import app


def _record(filename: str = "ok.docx") -> DocumentRecord:
    return DocumentRecord(
        id="abc123",
        filename=filename,
        content_type="application/octet-stream",
        num_chunks=1,
        num_chars=42,
        uploaded_at="2026-01-01T00:00:00+00:00",
    )


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "uploads").mkdir()

    settings = Settings(
        data_dir=data_dir,
        max_upload_bytes=1024,
        rerank_enabled=False,
        query_rewrite_enabled=False,
    )
    orch = MagicMock()
    orch.ingest.return_value = IngestResult(record=_record())
    orch.registry.count = 0
    orch.store.num_chunks = 0

    monkeypatch.setattr("app.api.routes.get_settings", lambda: settings)
    # Prevent lifespan from loading real embedding / FAISS models.
    monkeypatch.setattr("app.main.get_orchestrator", lambda: orch)
    app.dependency_overrides[get_orchestrator] = lambda: orch

    with TestClient(app, raise_server_exceptions=True) as c:
        c._orch = orch  # type: ignore[attr-defined]
        yield c

    app.dependency_overrides.clear()
    get_settings.cache_clear()


def test_upload_rejects_unsupported_extension(client: TestClient):
    res = client.post(
        "/documents/upload",
        files={"file": ("sheet.xlsx", b"a,b\n1,2\n", "application/vnd.ms-excel")},
    )
    assert res.status_code == 400
    assert "Unsupported" in res.json()["detail"]
    assert "DOCX" in res.json()["detail"]
    client._orch.ingest.assert_not_called()  # type: ignore[attr-defined]


def test_upload_rejects_empty_file(client: TestClient):
    res = client.post(
        "/documents/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


def test_upload_rejects_oversized_file(client: TestClient):
    payload = b"x" * 2000  # over fixture max_upload_bytes=1024
    res = client.post(
        "/documents/upload",
        files={"file": ("big.txt", payload, "text/plain")},
    )
    assert res.status_code == 413
    assert "too large" in res.json()["detail"].lower()
    client._orch.ingest.assert_not_called()  # type: ignore[attr-defined]


def test_upload_maps_corrupt_document_error(client: TestClient):
    client._orch.ingest.side_effect = CorruptedDocumentError(  # type: ignore[attr-defined]
        "DOCX 'broken.docx' is not a valid Word file (or is corrupted)."
    )
    res = client.post(
        "/documents/upload",
        files={
            "file": (
                "broken.docx",
                b"PK fake zip header content here",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert res.status_code == 422
    assert "corrupted" in res.json()["detail"].lower()


def test_upload_maps_empty_document_error(client: TestClient):
    client._orch.ingest.side_effect = EmptyDocumentError(  # type: ignore[attr-defined]
        "No extractable text found in 'scan.pdf'."
    )
    res = client.post(
        "/documents/upload",
        files={"file": ("scan.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert res.status_code == 422
    assert "extractable text" in res.json()["detail"].lower()


def test_upload_accepts_docx_when_ingest_ok(client: TestClient):
    client._orch.ingest.return_value = IngestResult(  # type: ignore[attr-defined]
        record=_record("notes.docx")
    )
    res = client.post(
        "/documents/upload",
        files={
            "file": (
                "notes.docx",
                b"fake-docx-bytes-but-ingest-mocked",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["document"]["filename"] == "notes.docx"
    client._orch.ingest.assert_called_once()  # type: ignore[attr-defined]
