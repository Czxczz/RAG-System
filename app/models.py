"""Pydantic request/response schemas for the public API."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    id: str
    filename: str
    content_type: str
    num_chunks: int
    num_chars: int
    uploaded_at: str


class UploadResponse(BaseModel):
    document: DocumentInfo
    message: str = "Document ingested successfully."


class DocumentList(BaseModel):
    documents: list[DocumentInfo]
    total: int


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language question.")
    mode: Literal["auto", "openai", "gemini", "ollama", "extractive"] = Field(
        default="auto",
        description=(
            "LLM routing mode. 'ollama' = local/privacy, "
            "'openai'/'gemini' = cloud."
        ),
    )
    top_k: Optional[int] = Field(
        default=3, ge=1, le=20, description="Override number of chunks to retrieve."
    )
    engine: Literal["custom", "langchain", "langgraph"] = Field(
        default="custom",
        description=(
            "Pipeline implementation: 'custom' = built-in orchestrator (default), "
            "'langchain' = LCEL wrapper, 'langgraph' = LangGraph node graph."
        ),
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional conversation id for multi-turn chat history. Omit to start "
            "a new thread; reuse the id returned in the response to continue."
        ),
    )
    document_ids: Optional[list[str]] = Field(
        default=None,
        description=(
            "Optional list of document ids to search. Omit or null to search all "
            "uploaded documents."
        ),
    )


class ChatTurnModel(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ConversationResponse(BaseModel):
    id: str
    turns: list[ChatTurnModel]
    created_at: str
    updated_at: str


class Citation(BaseModel):
    marker: int = Field(..., description="Citation number referenced in the answer, e.g. [1].")
    document_id: str
    filename: str
    page: int | None = Field(
        default=None,
        description="1-based PDF page number when available; null for TXT/Markdown.",
    )
    chunk_id: str
    score: float
    snippet: str
    chunk_text: str = Field(
        ...,
        description="Full retrieved chunk text (snippet is a short preview).",
    )


class ChatResponse(BaseModel):
    answer: str
    grounded: bool = Field(
        ...,
        description=(
            "False when no relevant context was found, or when the answer "
            "failed post-generation citation validation."
        ),
    )
    provider: str = Field(..., description="LLM provider that produced the answer.")
    citations: list[Citation]
    validation_notes: list[str] = Field(
        default_factory=list,
        description="Warnings from the answer validation gate, if any.",
    )
    conversation_id: str = Field(
        ...,
        description="Conversation id — send on the next request to continue the thread.",
    )


class HealthResponse(BaseModel):
    status: str
    embedding_provider: str
    llm_provider: str
    documents: int
    chunks: int
    auth_enabled: bool = False


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class LoginResponse(BaseModel):
    token: str
    username: str
    role: Literal["admin", "user"]
    message: str = "Logged in."


class MeResponse(BaseModel):
    username: str
    role: Literal["admin", "user"]
    auth_enabled: bool


class AuthStatusResponse(BaseModel):
    auth_enabled: bool
    default_hint: str | None = None


class ConfigUpdateRequest(BaseModel):
    """Partial admin config update. Omitted / empty secrets are left unchanged."""

    openai_api_key: Optional[str] = None
    openai_chat_model: Optional[str] = None
    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = None
    ollama_base_url: Optional[str] = None
    ollama_model: Optional[str] = None
    llm_provider: Optional[str] = None
    chunk_size: Optional[int] = Field(default=None, ge=100, le=8000)
    chunk_overlap: Optional[int] = Field(default=None, ge=0, le=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    min_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    rerank_enabled: Optional[bool] = None
    retrieve_k: Optional[int] = Field(default=None, ge=1, le=100)
    mmr_enabled: Optional[bool] = None
    mmr_lambda: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    mmr_dedup_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    ocr_enabled: Optional[bool] = None
    ocr_language: Optional[str] = None
    ocr_dpi: Optional[int] = Field(default=None, ge=72, le=600)
    ocr_min_chars_per_page: Optional[int] = Field(default=None, ge=0, le=500)
    retrieval_gate_enabled: Optional[bool] = None
    retrieval_gate_min_score: Optional[float] = None
    answer_validation_enabled: Optional[bool] = None
    answer_validation_min_support: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    prompt_injection_enabled: Optional[bool] = None
    prompt_injection_block: Optional[bool] = None
    max_upload_bytes: Optional[int] = Field(default=None, ge=1024, le=500 * 1024 * 1024)


class ConfigResponse(BaseModel):
    config: dict
    message: str = "OK"
