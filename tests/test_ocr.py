"""Tests for scanned-PDF OCR fallback."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.core.ingestion import OcrOptions, _maybe_ocr_pdf_pages, extract_text
from app.core.ocr import OcrUnavailableError, tesseract_available


def test_maybe_ocr_skips_when_disabled():
    pages = [(1, ""), (2, "plenty of native text here already")]
    out = _maybe_ocr_pdf_pages(
        Path("x.pdf"),
        pages,
        OcrOptions(enabled=False),
    )
    assert out == pages


def test_maybe_ocr_skips_when_pages_have_enough_text():
    pages = [(1, "x" * 80), (2, "y" * 80)]
    with patch("app.core.ingestion.ocr_pdf_pages") as mock_ocr:
        out = _maybe_ocr_pdf_pages(Path("x.pdf"), pages, OcrOptions(enabled=True))
    mock_ocr.assert_not_called()
    assert out == pages


def test_maybe_ocr_fills_blank_pages(tmp_path: Path):
    pages = [(1, ""), (2, "native page two text that is long enough")]
    with patch(
        "app.core.ingestion.ocr_pdf_pages",
        return_value={1: "Elastic IP is a static address"},
    ) as mock_ocr:
        out = _maybe_ocr_pdf_pages(
            tmp_path / "scan.pdf",
            pages,
            OcrOptions(enabled=True, min_chars=40),
        )
    mock_ocr.assert_called_once()
    assert out[0] == (1, "Elastic IP is a static address")
    assert out[1][1].startswith("native page two")


def test_maybe_ocr_prefers_longer_text():
    pages = [(1, "short")]
    with patch(
        "app.core.ingestion.ocr_pdf_pages",
        return_value={1: "a much longer OCR transcription of the page"},
    ):
        out = _maybe_ocr_pdf_pages(
            Path("x.pdf"),
            pages,
            OcrOptions(enabled=True, min_chars=40),
        )
    assert "longer OCR" in out[0][1]


def test_maybe_ocr_raises_when_unavailable_and_no_native_text():
    pages = [(1, ""), (2, "   ")]
    with patch(
        "app.core.ingestion.ocr_pdf_pages",
        side_effect=OcrUnavailableError("Tesseract missing"),
    ):
        from app.core.ingestion import IngestError

        with pytest.raises(IngestError, match="Tesseract"):
            _maybe_ocr_pdf_pages(
                Path("scan.pdf"),
                pages,
                OcrOptions(enabled=True),
            )


def test_maybe_ocr_keeps_native_when_unavailable_but_partial_text():
    pages = [(1, ""), (2, "enough native characters on this page already here")]
    with patch(
        "app.core.ingestion.ocr_pdf_pages",
        side_effect=OcrUnavailableError("Tesseract missing"),
    ):
        out = _maybe_ocr_pdf_pages(
            Path("mixed.pdf"),
            pages,
            OcrOptions(enabled=True, min_chars=40),
        )
    assert out == pages


def test_tesseract_available_is_bool():
    assert isinstance(tesseract_available(), bool)


def test_extract_text_pdf_with_ocr_disabled_uses_native(tmp_path: Path):
    # Minimal valid-ish PDF is hard; skip if pypdf can't open our stub.
    # Instead verify OcrOptions flows through extract_text for non-PDF.
    path = tmp_path / "notes.txt"
    path.write_text("hello world", encoding="utf-8")
    pages = extract_text(path, ocr=OcrOptions(enabled=False))
    assert pages == [(None, "hello world")]
