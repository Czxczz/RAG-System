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
    engine: Literal["custom", "langchain"] = Field(
        default="custom",
        description=(
            "Pipeline implementation: 'custom' = built-in orchestrator (default), "
            "'langchain' = LCEL wrapper over the same components."
        ),
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional conversation id for multi-turn chat history. Omit to start "
            "a new thread; reuse the id returned in the response to continue."
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
    chunk_id: str
    score: float
    snippet: str


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
