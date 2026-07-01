"""Offline tests for LLMRouter streaming.

Only the extractive path is exercised here so the suite stays fully offline
(no OpenAI/Gemini/Ollama network calls). The cloud/local streamers share the
same fallback plumbing, which is covered indirectly via the extractive
terminal fallback.
"""
from __future__ import annotations

from app.config import Settings
from app.core.llm_router import LLMRouter

_CONTEXT = "[1] An Elastic IP address is static and public."
_USER = f"CONTEXT:\n{_CONTEXT}\n\nQUESTION: What is an Elastic IP?"


def test_generate_stream_extractive_yields_context_and_provider():
    router = LLMRouter(Settings(llm_provider="extractive"))
    chunks = list(router.generate_stream("system", _USER, mode="extractive"))

    assert chunks, "expected at least one streamed chunk"
    providers = {provider for _, provider in chunks}
    assert providers == {"extractive"}

    text = "".join(delta for delta, _ in chunks)
    assert _CONTEXT in text


def test_generate_stream_matches_generate_for_extractive():
    router = LLMRouter(Settings(llm_provider="extractive"))
    streamed = "".join(d for d, _ in router.generate_stream("system", _USER, "extractive"))
    full, provider = router.generate("system", _USER, "extractive")
    assert provider == "extractive"
    assert streamed == full
