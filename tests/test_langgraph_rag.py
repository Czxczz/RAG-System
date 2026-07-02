"""Tests for the LangGraph RAG path (offline fakes, no FAISS/network)."""
from __future__ import annotations

from app.chains.langgraph_rag import LangGraphRAG
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

    def has_llm(self, mode: str = "auto") -> bool:
        return True

    def generate(self, system: str, user: str, mode: str) -> tuple[str, str]:
        self.calls.append((system, user, mode))
        return self._answer, self._provider

    def generate_with_history(self, system: str, user: str, mode: str, history):
        self.calls.append((system, user, mode))
        return self._answer, self._provider


def _graph(hits, answer, settings=None, provider="gemini") -> tuple[LangGraphRAG, _FakeLLM]:
    settings = settings or Settings()
    engine = _FakeQueryEngine(hits)
    llm = _FakeLLM(answer, provider)
    return LangGraphRAG(settings=settings, query_engine=engine, llm=llm), llm


def test_answer_returns_supported_grounded_result():
    hits = [_hit("An Elastic IP address is static and public.", 0)]
    rag, llm = _graph(hits, answer="An Elastic IP is static [1].")
    result = rag.answer("What is an Elastic IP?", mode="gemini")
    assert result.grounded is True
    assert result.provider == "gemini"
    assert "[1]" in result.answer
    assert llm.calls and llm.calls[0][2] == "gemini"


def test_retrieval_gate_refuses_without_calling_llm():
    settings = Settings(retrieval_gate_enabled=True, retrieval_gate_min_score=3.0)
    hits = [_hit("weak match", 0, score=1.0)]
    rag, llm = _graph(hits, answer="should not be used", settings=settings)
    result = rag.answer("q", mode="gemini")
    assert result.answer == NOT_FOUND_MESSAGE
    assert result.provider == "none"
    assert result.grounded is False
    assert llm.calls == []


def test_empty_retrieval_refuses():
    rag, llm = _graph([], answer="x")
    result = rag.answer("q")
    assert result.answer == NOT_FOUND_MESSAGE
    assert llm.calls == []


def test_answer_validation_flags_unsupported_citation():
    settings = Settings(answer_validation_enabled=True, answer_validation_min_support=0.5)
    hits = [_hit("Spot Instances can be interrupted.", 0)]
    rag, _ = _graph(hits, answer="Spot hibernates always [2].", settings=settings)
    result = rag.answer("q", mode="gemini")
    assert result.validated is False
    assert result.grounded is False
    assert "may not be fully supported" in result.answer


def test_answer_returns_conversation_id_and_uses_history():
    settings = Settings(chat_memory_contextualize_use_llm=False)
    hits = [_hit("An Elastic IP address is static and public.", 0)]
    rag, llm = _graph(hits, answer="An Elastic IP is static [1].", settings=settings)
    first = rag.answer("What is an Elastic IP?", mode="gemini")
    assert first.conversation_id

    engine = rag.query_engine
    follow = rag.answer(
        "How do I release it?",
        mode="gemini",
        conversation_id=first.conversation_id,
    )
    assert follow.conversation_id == first.conversation_id
    assert engine.last_call is not None
    assert "release it" in engine.last_call[0]
    assert len(llm.calls) == 2
