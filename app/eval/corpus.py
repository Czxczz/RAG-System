"""Helpers for resolving multi-document eval corpora."""
from __future__ import annotations

from app.core.registry import DocumentRegistry


def resolve_document_ids(
    registry: DocumentRegistry, filenames: list[str] | None
) -> list[str] | None:
    """Map eval-case filenames to ingested document ids."""
    if not filenames:
        return None
    mapping = registry.ids_for_filenames(filenames)
    missing = [name for name in filenames if name not in mapping]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(
            f"Eval corpus missing document(s): {missing_list}. "
            "Ingest PDFs first (see scripts/ingest_eval_corpus.py)."
        )
    return [mapping[name] for name in filenames]
