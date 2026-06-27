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
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    openai_embedding_model: str = "text-embedding-3-small"

    # ── Chunking ─────────────────────────────────────────────
    chunk_size: int = 800
    chunk_overlap: int = 120

    # ── Retrieval ────────────────────────────────────────────
    top_k: int = 3
    min_score: float = 0.20
    rerank_enabled: bool = True
    retrieve_k: int = 20  # FAISS pool size before cross-encoder rerank
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Query rewriting (multi-query retrieval) ──────────────
    # Expand the user query into several variants, retrieve for each, and fuse
    # the candidate pools (max-cosine) to improve recall + precision.
    query_rewrite_enabled: bool = True
    query_rewrite_num_variants: int = 3   # extra variants beyond the original
    query_rewrite_use_llm: bool = True    # use an LLM when a provider is available

    # ── Diversity-aware reranking (MMR) ──────────────────────
    # Maximal Marginal Relevance trims near-duplicate chunks while preserving
    # relevance. lambda=1.0 -> pure relevance, 0.0 -> pure diversity.
    mmr_enabled: bool = True
    # Tuned on the EC2 user-guide eval (scripts/tune_mmr.py): lambda below 0.7
    # hurt precision/recall, so 0.6 + a 0.88 dedup cap gave the lowest
    # redundancy that still preserved baseline precision (0.925) and recall.
    mmr_lambda: float = 0.6
    # Hard cap on redundancy: a candidate whose cosine to an already-selected
    # chunk is >= this is dropped outright (kills near/exact duplicates that a
    # soft MMR penalty alone lets through). 1.0 disables the hard filter.
    mmr_dedup_threshold: float = 0.88

    # ── Context grouping ─────────────────────────────────────
    # Group selected chunks by source document in reading order (and drop
    # overlapping text between adjacent chunks) for a more coherent context.
    context_grouping_enabled: bool = True

    # ── LLM Router ───────────────────────────────────────────
    # "auto" | "openai" | "gemini" | "ollama" | "extractive"
    llm_provider: str = "auto"
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com"
    ollama_base_url: str = "http://localhost:11434"
    # Best default for 8 GB Apple Silicon: strong instruction-following + citations.
    # On 16 GB+ use qwen2.5:7b or llama3.1:8b; on 32 GB+ use qwen2.5:14b.
    ollama_model: str = "qwen2.5:3b"
    ollama_num_ctx: int = 8192  # context window; RAG sends ~5 chunks + system prompt
    ollama_timeout: float = 3000.0  # seconds; increase on slower / 8 GB machines

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
