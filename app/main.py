"""FastAPI application entrypoint for PrivateRAG AI Assistant.

Run locally:
    uvicorn app.main:app --reload

Then open http://localhost:8000/docs for interactive API docs.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import router
from app.dependencies import get_orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up embedding model + FAISS index so the first request is fast.
    get_orchestrator()
    yield


app = FastAPI(
    title="PrivateRAG AI Assistant",
    description=(
        "A private, hybrid RAG knowledge assistant. Upload documents and get "
        "grounded answers with citations, using local or cloud LLMs."
    ),
    version=__version__,
    lifespan=lifespan,
)

# Permissive CORS for the future frontend (v2). Tighten in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["system"])
def root() -> dict:
    return {
        "name": "PrivateRAG AI Assistant",
        "version": __version__,
        "docs": "/docs",
    }
