"""OCR helpers for scanned / image-only PDF pages.

Uses ``pypdfium2`` to rasterize pages and ``pytesseract`` (Tesseract) to
read text. Enabled by default; disable with ``OCR_ENABLED=false``.

System dependency: the ``tesseract`` binary must be on PATH
(``brew install tesseract`` on macOS, or the Docker image installs it).
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class OcrUnavailableError(RuntimeError):
    """Tesseract or PDF rasterizer is missing."""


def tesseract_available() -> bool:
    """Return True when the Tesseract binary can be invoked."""
    try:
        import pytesseract
        from pytesseract import TesseractNotFoundError
    except ImportError:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except TesseractNotFoundError:
        return False
    except Exception:  # noqa: BLE001
        return False


def ocr_pdf_pages(
    path: Path,
    page_numbers: list[int],
    *,
    dpi: int = 200,
    language: str = "eng",
) -> dict[int, str]:
    """OCR selected 1-based PDF pages. Returns ``{page_number: text}``."""
    if not page_numbers:
        return {}

    try:
        import pypdfium2 as pdfium
        import pytesseract
        from pytesseract import TesseractNotFoundError
    except ImportError as exc:
        raise OcrUnavailableError(
            "OCR requires 'pypdfium2' and 'pytesseract'. "
            "Install with: pip install pypdfium2 pytesseract"
        ) from exc

    if not tesseract_available():
        raise OcrUnavailableError(
            "Tesseract OCR is not installed or not on PATH. "
            "Install it (macOS: brew install tesseract; "
            "Debian/Ubuntu: apt install tesseract-ocr) "
            "or set OCR_ENABLED=false to skip scanned PDFs."
        )

    try:
        pdf = pdfium.PdfDocument(str(path))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not open PDF for OCR: {exc}") from exc

    scale = max(dpi, 72) / 72.0
    out: dict[int, str] = {}
    try:
        page_count = len(pdf)
        for page_no in page_numbers:
            if page_no < 1 or page_no > page_count:
                continue
            page = pdf[page_no - 1]
            try:
                bitmap = page.render(scale=scale)
                pil_image = bitmap.to_pil()
                text = pytesseract.image_to_string(pil_image, lang=language) or ""
                out[page_no] = text.strip()
            except TesseractNotFoundError as exc:
                raise OcrUnavailableError(
                    "Tesseract OCR is not installed or not on PATH. "
                    "Install it (macOS: brew install tesseract) "
                    "or set OCR_ENABLED=false."
                ) from exc
            except Exception as exc:  # noqa: BLE001
                logger.warning("OCR failed for %s page %s: %s", path.name, page_no, exc)
                out[page_no] = ""
            finally:
                page.close()
    finally:
        pdf.close()

    return out
