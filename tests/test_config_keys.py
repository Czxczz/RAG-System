"""Tests for LLM auto-routing with admin llm_provider preference."""
from __future__ import annotations

from app.config import Settings
from app.core.llm_router import LLMRouter


def test_auto_prefers_admin_llm_provider_gemini():
    settings = Settings(
        llm_provider="gemini",
        openai_api_key="sk-should-not-win",
        gemini_api_key="AQ.test-gemini",
    )
    router = LLMRouter(settings)
    assert router.resolve_provider("auto") == "gemini"


def test_explicit_mode_overrides_admin_provider():
    settings = Settings(
        llm_provider="gemini",
        openai_api_key="sk-test",
        gemini_api_key="AQ.test",
    )
    router = LLMRouter(settings)
    assert router.resolve_provider("openai") == "openai"


def test_save_overrides_skips_masked_secrets(tmp_path):
    from app.core.runtime_config import load_overrides, save_overrides

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    save_overrides(data_dir, {"gemini_api_key": "AQ.real-key", "top_k": 5})
    save_overrides(data_dir, {"gemini_api_key": "AQ••••ey", "top_k": 6})
    loaded = load_overrides(data_dir)
    assert loaded["gemini_api_key"] == "AQ.real-key"
    assert loaded["top_k"] == 6
