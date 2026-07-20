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
    # Reject uploads larger than this (bytes). Default 25 MB.
    max_upload_bytes: int = 25 * 1024 * 1024

    # ── Embeddings ───────────────────────────────────────────
    embedding_provider: str = "local"  # "local" | "openai"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    openai_embedding_model: str = "text-embedding-3-small"

    # ── Chunking ─────────────────────────────────────────────
    chunk_size: int = 800
    chunk_overlap: int = 120

    # ── OCR (scanned PDFs) ───────────────────────────────────
    # When a PDF page has little/no native text, rasterize + Tesseract OCR.
    # Requires the tesseract binary (included in the Docker image).
    ocr_enabled: bool = True
    ocr_language: str = "eng"
    ocr_dpi: int = 200
    ocr_min_chars_per_page: int = 40

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
    # Tuned on EC2 eval (scripts/tune_mmr.py): lambda below 0.6 hurt recall;
    # 0.85 dedup + 0.6 lambda balances redundancy vs precision better than 0.88.
    mmr_lambda: float = 0.6
    # Hard cap: skip chunks whose embedding cosine to a kept chunk >= this.
    # Required — soft MMR alone keeps near-duplicates when relevance is high.
    mmr_dedup_threshold: float = 0.85
    # After MMR, drop chunks with near-identical text (catches exact PDF boilerplate).
    text_dedupe_enabled: bool = True
    text_dedupe_jaccard: float = 0.85  # word-overlap threshold; 1.0 = exact text only

    # ── Context grouping ─────────────────────────────────────
    # Group selected chunks by source document in reading order (and drop
    # overlapping text between adjacent chunks) for a more coherent context.
    context_grouping_enabled: bool = True

    # ── Retrieval confidence gate ────────────────────────────
    # Deterministic refusal before the LLM is called: if the best retrieved
    # chunk's score is below the threshold, the corpus almost certainly does not
    # answer the question, so we refuse instead of risking a fabricated answer.
    # The threshold applies to the post-rerank score: cross-encoder logits
    # (relevant >~ 0) when rerank is on, or cosine similarity when it is off.
    retrieval_gate_enabled: bool = True
    retrieval_gate_min_score: float = 0.0

    # ── Answer validation gate ───────────────────────────────
    # After generation, verify the answer's [n] citations point to real chunks
    # and that enough of them are supported by the cited text. Unsupported or
    # out-of-range citations append a brief disclaimer and mark the answer as
    # not fully grounded (the answer text is otherwise preserved).
    answer_validation_enabled: bool = True
    answer_validation_min_support: float = 0.5

    # ── Prompt injection defense ─────────────────────────────
    # Quarantine user/document text in delimiters and harden the system prompt.
    # When block=True, high-signal jailbreak phrases refuse before the LLM call.
    prompt_injection_enabled: bool = True
    prompt_injection_block: bool = True

    # ── Chat memory (multi-turn history) ─────────────────────
    # Keeps recent user/assistant turns per conversation_id (in-process).
    # Used to contextualize follow-up retrieval queries and pass prior turns to
    # the LLM. Send conversation_id on /chat to continue a thread.
    chat_memory_enabled: bool = True
    chat_memory_max_turns: int = 10  # max stored messages (user + assistant)
    chat_memory_contextualize: bool = True  # rewrite follow-ups for retrieval
    chat_memory_contextualize_use_llm: bool = True  # LLM rewrite when available

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
