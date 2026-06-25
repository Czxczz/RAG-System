"""Embedding service.

Provides a single `embed_texts` / `embed_query` interface backed by either:

  * local  -> sentence-transformers (default; free, private, no network)
  * openai -> OpenAI embeddings API (cloud)

All vectors are L2-normalised so that inner-product search in FAISS is
equivalent to cosine similarity.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from app.config import Settings, get_settings


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return (vectors / norms).astype("float32")


class EmbeddingService:
    """Lazy-loading embedding backend selected by configuration."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.provider = settings.embedding_provider.lower()
        self._local_model = None
        self._openai_client = None
        self._dim: int | None = None

    # ── Backends ─────────────────────────────────────────────
    def _load_local(self):
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer

            self._local_model = SentenceTransformer(self.settings.local_embedding_model)
        return self._local_model

    def _load_openai(self):
        if self._openai_client is None:
            from openai import OpenAI

            if not self.settings.openai_api_key:
                raise RuntimeError(
                    "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is not set."
                )
            self._openai_client = OpenAI(api_key=self.settings.openai_api_key)
        return self._openai_client

    # ── Public API ───────────────────────────────────────────
    @property
    def dimension(self) -> int:
        """Embedding dimensionality (probes the model once if unknown)."""
        if self._dim is None:
            self._dim = int(self.embed_texts(["dimension probe"]).shape[1])
        return self._dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")

        if self.provider == "openai":
            client = self._load_openai()
            resp = client.embeddings.create(
                model=self.settings.openai_embedding_model,
                input=texts,
            )
            vectors = np.array([d.embedding for d in resp.data], dtype="float32")
        else:  # local (default)
            model = self._load_local()
            vectors = np.asarray(
                model.encode(texts, show_progress_bar=False, convert_to_numpy=True),
                dtype="float32",
            )

        return _normalize(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query, returning a 1-D float32 vector."""
        return self.embed_texts([text])[0]


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(get_settings())
