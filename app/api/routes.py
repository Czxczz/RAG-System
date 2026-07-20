"""HTTP API routes."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.core.ingestion import (
    SUPPORTED_EXTENSIONS,
    SUPPORTED_FORMATS_LABEL,
    CorruptedDocumentError,
    EmptyDocumentError,
    EncryptedDocumentError,
    IngestError,
    UnsupportedFormatError,
)
from app.core.orchestrator import RAGOrchestrator
from app.dependencies import get_conversation_store, get_orchestrator
from app.models import (
    ChatRequest,
    ChatResponse,
    ChatTurnModel,
    Citation,
    ConversationResponse,
    DocumentInfo,
    DocumentList,
    HealthResponse,
    UploadResponse,
)

router = APIRouter()


def _to_citations(hits) -> list[Citation]:
    return [
        Citation(
            marker=i,
            document_id=hit.chunk.document_id,
            filename=hit.chunk.filename,
            page=hit.chunk.page,
            chunk_id=hit.chunk.id,
            score=round(hit.score, 4),
            snippet=hit.chunk.text[:280] + ("…" if len(hit.chunk.text) > 280 else ""),
            chunk_text=hit.chunk.text,
        )
        for i, hit in enumerate(hits, start=1)
    ]


def _to_chat_response(result) -> ChatResponse:
    return ChatResponse(
        answer=result.answer,
        grounded=result.grounded,
        provider=result.provider,
        citations=_to_citations(result.hits),
        validation_notes=result.validation_notes,
        conversation_id=result.conversation_id,
    )


def _to_info(record) -> DocumentInfo:
    return DocumentInfo(
        id=record.id,
        filename=record.filename,
        content_type=record.content_type,
        num_chunks=record.num_chunks,
        num_chars=record.num_chars,
        uploaded_at=record.uploaded_at,
    )


def _chat_kwargs(request: ChatRequest, orch: RAGOrchestrator) -> dict:
    document_ids = request.document_ids
    if document_ids:
        orch.normalize_document_scope(document_ids)
    return {
        "query": request.query,
        "mode": request.mode,
        "top_k": request.top_k,
        "conversation_id": request.conversation_id,
        "document_ids": document_ids,
    }


def _format_size(num_bytes: int) -> str:
    mb = num_bytes / (1024 * 1024)
    if mb >= 1:
        return f"{mb:.0f} MB" if mb >= 10 else f"{mb:.1f} MB"
    return f"{num_bytes // 1024} KB"


def _save_upload_capped(upload: UploadFile, dest: Path, max_bytes: int) -> int:
    """Stream upload to disk; raise HTTP 413 if larger than ``max_bytes``."""
    written = 0
    chunk_size = 1024 * 1024  # 1 MB
    with dest.open("wb") as out:
        while True:
            chunk = upload.file.read(chunk_size)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"File is too large. Maximum upload size is "
                        f"{_format_size(max_bytes)}."
                    ),
                )
            out.write(chunk)
    if written == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )
    return written


def _http_for_ingest_error(exc: Exception) -> HTTPException:
    """Map ingestion failures to clear client-facing HTTP errors."""
    if isinstance(exc, UnsupportedFormatError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, EncryptedDocumentError):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, CorruptedDocumentError):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, EmptyDocumentError):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, IngestError):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(
        status_code=500,
        detail=f"Failed to ingest document: {exc}",
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
                f"Unsupported file type '{ext or '(none)'}'. "
                f"Supported: {SUPPORTED_FORMATS_LABEL}."
            ),
        )

    settings = get_settings()
    dest = settings.uploads_dir / f"{uuid.uuid4().hex}{ext}"
    try:
        try:
            _save_upload_capped(file, dest, settings.max_upload_bytes)
        finally:
            await file.close()

        try:
            result = orch.ingest(
                dest,
                filename=filename,
                content_type=file.content_type or "application/octet-stream",
            )
        except Exception as exc:  # noqa: BLE001 — mapped to HTTP below
            dest.unlink(missing_ok=True)
            raise _http_for_ingest_error(exc) from exc
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise

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
    try:
        kwargs = _chat_kwargs(request, orch)
        if request.engine == "langchain":
            from app.dependencies import get_langchain_rag

            result = get_langchain_rag().answer(**kwargs)
        elif request.engine == "langgraph":
            from app.dependencies import get_langgraph_rag

            result = get_langgraph_rag().answer(**kwargs)
        else:
            result = orch.answer(**kwargs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _to_chat_response(result)


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationResponse,
    tags=["chat"],
)
def get_conversation(conversation_id: str) -> ConversationResponse:
    conv = get_conversation_store().get(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return ConversationResponse(
        id=conv.id,
        turns=[ChatTurnModel(role=t.role, content=t.content) for t in conv.turns],
        created_at=conv.created_at,
        updated_at=conv.updated_at,
    )


@router.delete("/conversations/{conversation_id}", tags=["chat"])
def delete_conversation(conversation_id: str) -> dict:
    if not get_conversation_store().delete(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"message": "Conversation deleted.", "conversation_id": conversation_id}


@router.post("/chat/stream", tags=["chat"])
def chat_stream(
    request: ChatRequest, orch: RAGOrchestrator = Depends(get_orchestrator)
) -> StreamingResponse:
    """Stream a grounded answer as Server-Sent Events (LangChain engine).

    Emits ``data: {...}`` lines with ``type`` of ``citations`` (sources, sent
    first), ``token`` (incremental answer deltas), and ``done`` (final
    grounded/provider/validation status). Streaming always uses the LangChain
    (LCEL) path, which reuses the same FAISS index, retrieval gate, and answer
    validation gate as the custom engine.
    """
    from app.dependencies import get_langchain_rag

    rag = get_langchain_rag()

    try:
        kwargs = _chat_kwargs(request, orch)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def event_stream():
        try:
            for event in rag.stream(**kwargs):
                yield f"data: {json.dumps(event)}\n\n"
        except ValueError as exc:
            payload = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
