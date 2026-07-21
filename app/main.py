"""FastAPI application entrypoint for PrivateRAG AI Assistant.

Copyright (c) 2026 PrivateRAG. All rights reserved.

Run locally:
    uvicorn app.main:app --reload

Then open http://localhost:8000/ for the Web UI (or /docs for API docs).
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __copyright__, __version__
from app.api.routes import router
from app.dependencies import get_orchestrator

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up embedding model + FAISS index so the first request is fast.
    get_orchestrator()
    yield


app = FastAPI(
    title="PrivateRAG AI Assistant",
    description=(
        "A private, hybrid RAG knowledge assistant. Upload documents and get "
        "grounded answers with citations, using local or cloud LLMs.\n\n"
        f"{__copyright__} Licensed for single internal use; see LICENSE. "
        "Buyers pay their own cloud hosting and LLM API / token costs."
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

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", tags=["system"], include_in_schema=False)
def ui() -> FileResponse:
    """Serve the Web UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api", tags=["system"])
def root() -> dict:
    return {
        "name": "PrivateRAG AI Assistant",
        "version": __version__,
        "copyright": __copyright__,
        "docs": "/docs",
    }
