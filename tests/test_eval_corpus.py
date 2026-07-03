"""Tests for eval corpus filename resolution."""
from __future__ import annotations

import pytest

from app.core.registry import DocumentRecord, DocumentRegistry
from app.eval.corpus import resolve_document_ids


def test_resolve_document_ids_maps_filenames(tmp_path):
    registry = DocumentRegistry(tmp_path / "documents.json")
    registry.add(
        DocumentRecord(
            id="abc123",
            filename="ec2-ug.pdf",
            content_type="application/pdf",
            num_chunks=10,
            num_chars=1000,
            uploaded_at="2026-01-01T00:00:00+00:00",
        )
    )

    ids = resolve_document_ids(registry, ["ec2-ug.pdf"])
    assert ids == ["abc123"]


def test_resolve_document_ids_raises_when_missing(tmp_path):
    registry = DocumentRegistry(tmp_path / "documents.json")

    with pytest.raises(ValueError, match="ec2-instance-types.pdf"):
        resolve_document_ids(registry, ["ec2-instance-types.pdf"])
