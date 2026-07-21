"""Ingestion pipeline.

Transforms a raw document into clean, fixed-size text chunks:

    load -> extract text (-> OCR for scanned PDF pages) -> clean -> chunk

Supported formats: PDF (.pdf, text or scanned with OCR), Word (.docx),
Markdown (.md/.markdown), plain text (.txt).
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ProgressCallback = Callable[[str, int], None]  # (stage, percent 0-100)

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.ocr import OcrUnavailableError, ocr_pdf_pages

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown", ".txt"}

# Human-readable labels for API / UI error messages.
SUPPORTED_FORMATS_LABEL = "PDF (text or scanned/OCR), DOCX, TXT, Markdown"


class IngestError(ValueError):
    """Base class for user-facing ingestion failures."""


class UnsupportedFormatError(IngestError):
    """File extension is not in the allowlist."""


class EncryptedDocumentError(IngestError):
    """Document is password-protected and cannot be read."""


class CorruptedDocumentError(IngestError):
    """File is damaged or not a valid document of its claimed type."""


class EmptyDocumentError(IngestError):
    """No extractable text (blank file, or scanned PDF with OCR disabled)."""


@dataclass
class Chunk:
    """A single retrievable unit of text with provenance metadata."""

    text: str
    chunk_index: int
    page: int | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class OcrOptions:
    """Optional OCR settings for scanned PDF pages."""

    enabled: bool = True
    language: str = "eng"
    dpi: int = 200
    # If native text on a page has fewer characters than this, try OCR.
    min_chars: int = 40


# ─────────────────────────────────────────────────────────────
# 1. Text extraction
# ─────────────────────────────────────────────────────────────
def extract_text(
    path: Path,
    ocr: OcrOptions | None = None,
) -> list[tuple[int | None, str]]:
    """Extract text from a document.

    Returns a list of (page_number, text) tuples. For non-paginated formats
    (txt, md, docx) the page number is None and there is a single entry.
    """
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported file type '{suffix}'. "
            f"Supported: {SUPPORTED_FORMATS_LABEL}."
        )

    if suffix == ".pdf":
        return _extract_pdf(path, ocr=ocr or OcrOptions())
    if suffix == ".docx":
        return _extract_docx(path)

    # md / txt
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        raise CorruptedDocumentError(
            f"Could not read '{path.name}': {exc}"
        ) from exc
    return [(None, text)]


def _extract_pdf(
    path: Path,
    ocr: OcrOptions,
) -> list[tuple[int | None, str]]:
    try:
        reader = PdfReader(str(path))
    except PdfReadError as exc:
        raise CorruptedDocumentError(
            f"PDF '{path.name}' appears corrupted or unreadable. "
            "Try re-exporting or converting it to a new PDF."
        ) from exc
    except Exception as exc:  # noqa: BLE001 — surface as corrupt to the client
        raise CorruptedDocumentError(
            f"Failed to open PDF '{path.name}': {exc}"
        ) from exc

    if getattr(reader, "is_encrypted", False):
        unlocked = False
        try:
            # Empty password unlocks some "encrypted" but open PDFs.
            result = reader.decrypt("")  # type: ignore[attr-defined]
            unlocked = bool(result)
        except Exception:
            unlocked = False
        if not unlocked:
            raise EncryptedDocumentError(
                f"PDF '{path.name}' is password-protected. "
                "Remove the password and upload again."
            )

    pages: list[tuple[int | None, str]] = []
    try:
        for i, page in enumerate(reader.pages, start=1):
            pages.append((i, page.extract_text() or ""))
    except Exception as exc:  # noqa: BLE001
        raise CorruptedDocumentError(
            f"PDF '{path.name}' could not be parsed: {exc}"
        ) from exc

    return _maybe_ocr_pdf_pages(path, pages, ocr)


def _maybe_ocr_pdf_pages(
    path: Path,
    pages: list[tuple[int | None, str]],
    ocr: OcrOptions,
) -> list[tuple[int | None, str]]:
    """Fill in low-text / blank PDF pages via OCR when enabled."""
    if not ocr.enabled or not pages:
        return pages

    weak_pages = [
        int(page_no)
        for page_no, text in pages
        if page_no is not None and len((text or "").strip()) < ocr.min_chars
    ]
    if not weak_pages:
        return pages

    try:
        ocr_texts = ocr_pdf_pages(
            path,
            weak_pages,
            dpi=ocr.dpi,
            language=ocr.language,
        )
    except OcrUnavailableError as exc:
        # If native text already exists somewhere, keep it; otherwise fail clearly.
        if any((text or "").strip() for _, text in pages):
            logger.warning(
                "OCR unavailable for %s (%s); using native text only.",
                path.name,
                exc,
            )
            return pages
        raise IngestError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        if any((text or "").strip() for _, text in pages):
            logger.warning("OCR failed for %s: %s", path.name, exc)
            return pages
        raise IngestError(
            f"OCR failed for scanned PDF '{path.name}': {exc}"
        ) from exc

    merged: list[tuple[int | None, str]] = []
    for page_no, text in pages:
        native = (text or "").strip()
        if page_no is not None and page_no in ocr_texts:
            ocr_text = (ocr_texts[page_no] or "").strip()
            # Prefer the longer of native vs OCR for that page.
            if len(ocr_text) > len(native):
                merged.append((page_no, ocr_text))
                continue
        merged.append((page_no, text))
    return merged


def _extract_docx(path: Path) -> list[tuple[int | None, str]]:
    try:
        from docx import Document
        from docx.opc.exceptions import PackageNotFoundError
    except ImportError as exc:  # pragma: no cover
        raise IngestError(
            "DOCX support requires the 'python-docx' package. "
            "Install with: pip install python-docx"
        ) from exc

    try:
        document = Document(str(path))
    except PackageNotFoundError as exc:
        raise CorruptedDocumentError(
            f"DOCX '{path.name}' is not a valid Word file (or is corrupted)."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).lower()
        if "password" in msg or "encrypt" in msg:
            raise EncryptedDocumentError(
                f"DOCX '{path.name}' is password-protected. "
                "Remove the password and upload again."
            ) from exc
        raise CorruptedDocumentError(
            f"Failed to open DOCX '{path.name}': {exc}"
        ) from exc

    parts: list[str] = []
    for para in document.paragraphs:
        text = (para.text or "").strip()
        if text:
            parts.append(text)

    # Include table cell text so spreadsheets-in-Word aren't dropped.
    for table in document.tables:
        for row in table.rows:
            cells = [
                (cell.text or "").strip()
                for cell in row.cells
                if (cell.text or "").strip()
            ]
            if cells:
                parts.append(" | ".join(cells))

    return [(None, "\n\n".join(parts))]


# ─────────────────────────────────────────────────────────────
# 2. Cleaning
# ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Normalise whitespace and strip common noise."""
    # De-hyphenate words split across line breaks: "exam-\nple" -> "example"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Collapse Windows newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip trailing spaces on each line
    text = re.sub(r"[ \t]+\n", "\n", text)
    # Collapse 3+ blank lines into a paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces/tabs
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────
# 3. Chunking
# ─────────────────────────────────────────────────────────────
def _split_sentences(text: str) -> list[str]:
    """Lightweight sentence/paragraph splitter (no heavy NLP deps)."""
    # Split on paragraph breaks first, then sentence boundaries.
    pieces: list[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", para)
        for s in sentences:
            s = s.strip()
            if s:
                pieces.append(s)
    return pieces


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    page: int | None = None,
    start_index: int = 0,
) -> list[Chunk]:
    """Greedily pack sentences into ~chunk_size-character chunks with overlap.

    Overlap is achieved by carrying trailing sentences from the previous chunk
    into the next one, preserving semantic continuity across boundaries.
    """
    sentences = _split_sentences(text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    idx = start_index

    def flush() -> list[str]:
        nonlocal current, current_len, idx
        if not current:
            return []
        chunk_str = " ".join(current).strip()
        if chunk_str:
            chunks.append(Chunk(text=chunk_str, chunk_index=idx, page=page))
            idx += 1
        # Build overlap tail from the end of the current chunk.
        tail: list[str] = []
        tail_len = 0
        for sent in reversed(current):
            if tail_len + len(sent) > chunk_overlap:
                break
            tail.insert(0, sent)
            tail_len += len(sent) + 1
        return tail

    for sentence in sentences:
        # A single oversized sentence is hard-split by characters.
        if len(sentence) > chunk_size:
            if current:
                current = flush()
                current_len = sum(len(s) + 1 for s in current)
            for i in range(0, len(sentence), chunk_size - chunk_overlap):
                piece = sentence[i : i + chunk_size]
                chunks.append(Chunk(text=piece, chunk_index=idx, page=page))
                idx += 1
            continue

        if current_len + len(sentence) + 1 > chunk_size and current:
            current = flush()
            current_len = sum(len(s) + 1 for s in current)

        current.append(sentence)
        current_len += len(sentence) + 1

    if current:
        flush()

    return chunks


def ingest_file(
    path: Path,
    chunk_size: int,
    chunk_overlap: int,
    ocr: OcrOptions | None = None,
    on_progress: ProgressCallback | None = None,
) -> list[Chunk]:
    """Full pipeline for one file: extract -> clean -> chunk."""
    def report(stage: str, percent: int) -> None:
        if on_progress:
            on_progress(stage, percent)

    report("extracting", 20)
    pages = extract_text(path, ocr=ocr)
    report("chunking", 50)
    chunks: list[Chunk] = []
    next_index = 0
    for page_no, raw in pages:
        cleaned = clean_text(raw)
        if not cleaned:
            continue
        page_chunks = chunk_text(
            cleaned,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            page=page_no,
            start_index=next_index,
        )
        chunks.extend(page_chunks)
        next_index += len(page_chunks)

    if not chunks:
        hint = (
            "If this is a scanned PDF, enable OCR (OCR_ENABLED=true) and install "
            "Tesseract, or use a text-based PDF / DOCX / TXT / Markdown file."
        )
        raise EmptyDocumentError(
            f"No extractable text found in '{path.name}'. {hint}"
        )
    report("chunked", 60)
    return chunks
