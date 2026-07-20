# PrivateRAG — production image
# Build:  docker compose build
# Run:    docker compose up
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Persist HF / sentence-transformers downloads under /cache (compose volume).
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/cache/sentence-transformers

WORKDIR /app

# System libs: faiss/sentence-transformers build deps + Tesseract OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        tesseract-ocr \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts

RUN mkdir -p /app/data /cache

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

# First boot downloads local embedding/reranker models (~a few hundred MB).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
