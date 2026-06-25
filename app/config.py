"""Application configuration.

All settings are environment-driven (12-factor) with safe defaults so the
system runs out-of-the-box without any API keys or external services.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Storage ──────────────────────────────────────────────
    data_dir: Path = Path("data")

    # ── Embeddings ───────────────────────────────────────────
    embedding_provider: str = "local"  # "local" | "openai"
    local_embedding_model: str = "all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"

    # ── Chunking ─────────────────────────────────────────────
    chunk_size: int = 800
    chunk_overlap: int = 120

    # ── Retrieval ────────────────────────────────────────────
    top_k: int = 5
    min_score: float = 0.20
    rerank_enabled: bool = True
    retrieve_k: int = 20  # FAISS pool size before cross-encoder rerank
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── LLM Router ───────────────────────────────────────────
    # "auto" | "openai" | "gemini" | "ollama" | "extractive"
    llm_provider: str = "auto"
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    # ── Derived paths ────────────────────────────────────────
    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def index_path(self) -> Path:
        return self.data_dir / "faiss.index"

    @property
    def metadata_path(self) -> Path:
        return self.data_dir / "metadata.json"

    @property
    def documents_path(self) -> Path:
        return self.data_dir / "documents.json"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
