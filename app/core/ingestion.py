"""Ingestion pipeline.

Transforms a raw document into clean, semantically-sized text chunks:

    load -> extract text -> clean -> chunk

Supported formats: PDF (.pdf), Markdown (.md/.markdown), plain text (.txt).
Embedding + storage happen downstream in the orchestrator.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}


@dataclass
class Chunk:
    """A single retrievable unit of text with provenance metadata."""

    text: str
    chunk_index: int
    page: int | None = None
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# 1. Text extraction
# ─────────────────────────────────────────────────────────────
def extract_text(path: Path) -> list[tuple[int | None, str]]:
    """Extract text from a document.

    Returns a list of (page_number, text) tuples. For non-paginated formats
    (txt, md) the page number is None and there is a single entry.
    """
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages: list[tuple[int | None, str]] = []
        for i, page in enumerate(reader.pages, start=1):
            pages.append((i, page.extract_text() or ""))
        return pages

    # md / txt
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [(None, text)]


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


def ingest_file(path: Path, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Full pipeline for one file: extract -> clean -> chunk."""
    pages = extract_text(path)
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
    return chunks
