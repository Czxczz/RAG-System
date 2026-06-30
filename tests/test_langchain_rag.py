"""Tests for the LangChain (LCEL) RAG wrapper.

These use lightweight fakes for the query engine and LLM router so they run
offline (no FAISS, no network) and assert parity with the custom orchestrator's
behaviour: retrieval gate, marker-aligned citations, and answer validation.
"""
from __future__ import annotations

from app.chains.langchain_rag import LangChainRAG, QueryEngineRetriever, hit_to_document
from app.config import Settings
from app.core.orchestrator import NOT_FOUND_MESSAGE
from app.core.vector_store import SearchHit, StoredChunk


def _hit(text: str, chunk_index: int = 0, score: float = 5.0, page: int = 1) -> SearchHit:
    return SearchHit(
        chunk=StoredChunk(
            id=f"doc:{chunk_index}",
            document_id="doc",
            filename="ec2-ug.pdf",
            chunk_index=chunk_index,
            page=page,
            text=text,
        ),
        score=score,
    )


class _FakeQueryEngine:
    """Returns a fixed list of hits regardless of query."""

    def __init__(self, hits: list[SearchHit]) -> None:
        self._hits = hits
        self.last_call: tuple[str, int, float] | None = None

    def retrieve(self, query: str, top_k: int, min_score: float) -> list[SearchHit]:
        self.last_call = (query, top_k, min_score)
        return self._hits[:top_k]


class _FakeLLM:
    def __init__(self, answer: str, provider: str = "gemini") -> None:
        self._answer = answer
        self._provider = provider
        self.calls: list[tuple[str, str, str]] = []

    def generate(self, system: str, user: str, mode: str) -> tuple[str, str]:
        self.calls.append((system, user, mode))
        return self._answer, self._provider


def _rag(hits, answer, settings=None, provider="gemini") -> tuple[LangChainRAG, _FakeLLM]:
    settings = settings or Settings()
    engine = _FakeQueryEngine(hits)
    llm = _FakeLLM(answer, provider)
    return LangChainRAG(settings=settings, query_engine=engine, llm=llm), llm


# ── Document conversion ──────────────────────────────────────
def test_hit_to_document_carries_citation_metadata():
    doc = hit_to_document(_hit("Elastic IP is static.", 2, score=6.1, page=42), marker=3)
    assert doc.page_content == "Elastic IP is static."
    assert doc.metadata["marker"] == 3
    assert doc.metadata["filename"] == "ec2-ug.pdf"
    assert doc.metadata["page"] == 42
    assert doc.metadata["score"] == 6.1


def test_retriever_returns_marker_aligned_documents():
    hits = [_hit("first", 0), _hit("second", 1)]
    retriever = QueryEngineRetriever(
        query_engine=_FakeQueryEngine(hits), top_k=3, min_score=0.2
    )
    docs = retriever.invoke("anything")
    assert [d.metadata["marker"] for d in docs] == [1, 2]
    assert docs[0].page_content == "first"


# ── Answer flow ──────────────────────────────────────────────
def test_answer_returns_supported_grounded_result():
    hits = [_hit("An Elastic IP address is static and public.", 0)]
    rag, llm = _rag(hits, answer="An Elastic IP is static [1].")
    result = rag.answer("What is an Elastic IP?", mode="gemini")
    assert result.grounded is True
    assert result.provider == "gemini"
    assert "[1]" in result.answer
    assert llm.calls and llm.calls[0][2] == "gemini"  # mode threaded through


def test_retrieval_gate_refuses_without_calling_llm():
    settings = Settings(retrieval_gate_enabled=True, retrieval_gate_min_score=3.0)
    hits = [_hit("weak match", 0, score=1.0)]
    rag, llm = _rag(hits, answer="should not be used", settings=settings)
    result = rag.answer("q", mode="gemini")
    assert result.answer == NOT_FOUND_MESSAGE
    assert result.provider == "none"
    assert result.grounded is False
    assert llm.calls == []  # LLM never invoked


def test_empty_retrieval_refuses():
    rag, llm = _rag([], answer="x")
    result = rag.answer("q")
    assert result.answer == NOT_FOUND_MESSAGE
    assert llm.calls == []


def test_answer_validation_flags_unsupported_citation():
    settings = Settings(answer_validation_enabled=True, answer_validation_min_support=0.5)
    hits = [_hit("Spot Instances can be interrupted.", 0)]
    # Cites [2] which does not exist -> validation fails, disclaimer appended.
    rag, _ = _rag(hits, answer="Spot hibernates always [2].", settings=settings)
    result = rag.answer("q", mode="gemini")
    assert result.validated is False
    assert result.grounded is False
    assert "may not be fully supported" in result.answer


def test_as_runnable_invocation():
    hits = [_hit("Security groups control inbound traffic.", 0)]
    rag, _ = _rag(hits, answer="Configure security groups [1].")
    chain = rag.as_runnable()
    result = chain.invoke({"query": "inbound?", "mode": "gemini", "top_k": 3})
    assert result.grounded is True
    assert "[1]" in result.answer
