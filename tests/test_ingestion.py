"""Unit tests for the ingestion pipeline (no heavy deps required)."""
from app.core.ingestion import chunk_text, clean_text


def test_clean_text_dehyphenates_and_collapses():
    raw = "This is an exam-\nple   of   noisy\r\n\n\n\ntext."
    cleaned = clean_text(raw)
    assert "example" in cleaned
    assert "  " not in cleaned
    assert "\n\n\n" not in cleaned


def test_chunking_respects_size_and_overlap():
    text = " ".join(f"Sentence number {i}." for i in range(200))
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    # Each chunk should be roughly within the target size (allow some slack).
    assert all(len(c.text) <= 260 for c in chunks)
    # Chunk indices are sequential.
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_oversized_sentence_is_hard_split():
    text = "x" * 1000
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
