"""HTTP API routes."""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.config import get_settings
from app.core.ingestion import SUPPORTED_EXTENSIONS
from app.core.orchestrator import RAGOrchestrator
from app.dependencies import get_orchestrator
from app.models import (
    ChatRequest,
    ChatResponse,
    Citation,
    DocumentInfo,
    DocumentList,
    HealthResponse,
    UploadResponse,
)

router = APIRouter()


def _to_info(record) -> DocumentInfo:
    return DocumentInfo(
        id=record.id,
        filename=record.filename,
        content_type=record.content_type,
        num_chunks=record.num_chunks,
        num_chars=record.num_chars,
        uploaded_at=record.uploaded_at,
    )


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(orch: RAGOrchestrator = Depends(get_orchestrator)) -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        embedding_provider=settings.embedding_provider,
        llm_provider=settings.llm_provider,
        documents=orch.registry.count,
        chunks=orch.store.num_chunks,
    )


@router.post("/documents/upload", response_model=UploadResponse, tags=["documents"])
async def upload_document(
    file: UploadFile = File(...),
    orch: RAGOrchestrator = Depends(get_orchestrator),
) -> UploadResponse:
    filename = file.filename or "untitled"
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    settings = get_settings()
    dest = settings.uploads_dir / f"{uuid.uuid4().hex}{ext}"
    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    finally:
        await file.close()

    try:
        result = orch.ingest(
            dest, filename=filename, content_type=file.content_type or "application/octet-stream"
        )
    except ValueError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return UploadResponse(document=_to_info(result.record))


@router.get("/documents", response_model=DocumentList, tags=["documents"])
def list_documents(orch: RAGOrchestrator = Depends(get_orchestrator)) -> DocumentList:
    records = orch.registry.list()
    return DocumentList(documents=[_to_info(r) for r in records], total=len(records))


@router.delete("/documents/{document_id}", tags=["documents"])
def delete_document(
    document_id: str, orch: RAGOrchestrator = Depends(get_orchestrator)
) -> dict:
    if not orch.delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"message": "Document deleted.", "document_id": document_id}


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(
    request: ChatRequest, orch: RAGOrchestrator = Depends(get_orchestrator)
) -> ChatResponse:
    if request.engine == "langchain":
        from app.dependencies import get_langchain_rag

        result = get_langchain_rag().answer(
            query=request.query, mode=request.mode, top_k=request.top_k
        )
    else:
        result = orch.answer(query=request.query, mode=request.mode, top_k=request.top_k)
    citations = [
        Citation(
            marker=i,
            document_id=hit.chunk.document_id,
            filename=hit.chunk.filename,
            chunk_id=hit.chunk.id,
            score=round(hit.score, 4),
            snippet=hit.chunk.text[:280] + ("…" if len(hit.chunk.text) > 280 else ""),
        )
        for i, hit in enumerate(result.hits, start=1)
    ]
    return ChatResponse(
        answer=result.answer,
        grounded=result.grounded,
        provider=result.provider,
        citations=citations,
        validation_notes=result.validation_notes,
    )
