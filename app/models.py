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


class Citation(BaseModel):
    marker: int = Field(..., description="Citation number referenced in the answer, e.g. [1].")
    document_id: str
    filename: str
    chunk_id: str
    score: float
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    grounded: bool = Field(..., description="False when no relevant context was found.")
    provider: str = Field(..., description="LLM provider that produced the answer.")
    citations: list[Citation]


class HealthResponse(BaseModel):
    status: str
    embedding_provider: str
    llm_provider: str
    documents: int
    chunks: int
