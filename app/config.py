"""Application configuration.

All settings are environment-driven (12-factor) with safe defaults so the
system runs out-of-the-box without any API keys or external services.

Admin UI overrides are persisted in ``data/runtime_settings.json`` and merged
on top of ``.env`` / environment values.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.runtime_config import load_overrides


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Storage ──────────────────────────────────────────────
    data_dir: Path = Path("data")
    # Reject uploads larger than this (bytes). Default 25 MB.
    max_upload_bytes: int = 25 * 1024 * 1024

    # ── Auth (admin / user roles) ─────────────────────────────
    # When false, all API calls act as admin (zero-login local demo).
    auth_enabled: bool = False
    auth_secret: str = "change-me-privaterag-secret"
    auth_token_ttl_seconds: int = 60 * 60 * 24
    admin_username: str = "admin"
    admin_password: str = "admin"
    user_username: str = "user"
    user_password: str = "user"

    # ── Embeddings ───────────────────────────────────────────
    embedding_provider: str = "local"  # "local" | "openai"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    openai_embedding_model: str = "text-embedding-3-small"

    # ── Chunking ─────────────────────────────────────────────
    chunk_size: int = 800
    chunk_overlap: int = 120

    # ── OCR (scanned PDFs) ───────────────────────────────────
    ocr_enabled: bool = True
    ocr_language: str = "eng"
    ocr_dpi: int = 200
    ocr_min_chars_per_page: int = 40

    # ── Retrieval ────────────────────────────────────────────
    top_k: int = 5
    min_score: float = 0.20
    rerank_enabled: bool = True
    retrieve_k: int = 40
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Hybrid (dense + BM25) ────────────────────────────────
    hybrid_enabled: bool = True
    bm25_top_k: int = 40
    rrf_k: int = 60
    overview_demote_enabled: bool = True
    overview_demote_strength: float = 0.55

    # ── Query rewriting ──────────────────────────────────────
    query_rewrite_enabled: bool = True
    query_rewrite_num_variants: int = 3
    query_rewrite_use_llm: bool = True

    # ── MMR / dedupe ─────────────────────────────────────────
    mmr_enabled: bool = True
    mmr_lambda: float = 0.55
    mmr_dedup_threshold: float = 0.80
    text_dedupe_enabled: bool = True
    text_dedupe_jaccard: float = 0.85

    # ── Context grouping ─────────────────────────────────────
    context_grouping_enabled: bool = True

    # ── Retrieval confidence gate ────────────────────────────
    retrieval_gate_enabled: bool = True
    retrieval_gate_min_score: float = 0.0

    # ── Answer validation gate ───────────────────────────────
    answer_validation_enabled: bool = True
    answer_validation_min_support: float = 0.5

    # ── Prompt injection defense ─────────────────────────────
    prompt_injection_enabled: bool = True
    prompt_injection_block: bool = True

    # ── Chat memory ──────────────────────────────────────────
    chat_memory_enabled: bool = True
    chat_memory_max_turns: int = 10
    chat_memory_contextualize: bool = True
    chat_memory_contextualize_use_llm: bool = True

    # ── LLM Router ───────────────────────────────────────────
    llm_provider: str = "auto"
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_num_ctx: int = 8192
    ollama_timeout: float = 3000.0

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


def _build_settings() -> Settings:
    base = Settings()
    overrides = load_overrides(base.data_dir)
    if overrides:
        base = base.model_copy(update=overrides)
    base.ensure_dirs()
    return base


@lru_cache
def get_settings() -> Settings:
    return _build_settings()


def reload_settings() -> Settings:
    """Clear the settings cache and rebuild (after admin config save)."""
    get_settings.cache_clear()
    return get_settings()
