"""Tests for in-memory chat history and follow-up contextualization."""
from __future__ import annotations

from app.config import Settings
from app.core.conversation_memory import (
    ChatTurn,
    ConversationStore,
    contextualize_query,
    needs_contextualization,
)
from app.core.llm_router import LLMRouter


class _FakeLLM:
    def __init__(self, rewritten: str = "Elastic IP static public address") -> None:
        self._rewritten = rewritten
        self.calls: list[tuple[str, str, str]] = []

    def has_llm(self, mode: str = "auto") -> bool:
        return True

    def generate(self, system: str, user: str, mode: str) -> tuple[str, str]:
        self.calls.append((system, user, mode))
        return self._rewritten, "gemini"


def test_conversation_store_trims_to_max_turns():
    store = ConversationStore()
    conv_id, _ = store.get_or_create()
    store.append_exchange(conv_id, "q1", "a1", max_turns=4)
    store.append_exchange(conv_id, "q2", "a2", max_turns=4)
    conv = store.get(conv_id)
    assert conv is not None
    assert len(conv.turns) == 4
    assert conv.turns[-2].content == "q2"
    assert conv.turns[-1].content == "a2"


def test_needs_contextualization_for_referential_follow_up():
    history = [
        ChatTurn(role="user", content="What is an Elastic IP?"),
        ChatTurn(role="assistant", content="It is a static public address [1]."),
    ]
    assert needs_contextualization("How do I release it?", history) is True
    assert (
        needs_contextualization(
            "What are the detailed steps for configuring network interfaces?",
            history,
        )
        is False
    )


def test_heuristic_contextualize_when_llm_disabled():
    settings = Settings(
        chat_memory_contextualize=True,
        chat_memory_contextualize_use_llm=False,
    )
    history = [
        ChatTurn(role="user", content="What is an Elastic IP?"),
        ChatTurn(role="assistant", content="Static public IP [1]."),
    ]
    rewritten = contextualize_query("How do I release it?", history, settings, _FakeLLM())
    assert "Elastic IP" in rewritten
    assert "release it" in rewritten


def test_llm_contextualize_when_enabled():
    settings = Settings(
        chat_memory_contextualize=True,
        chat_memory_contextualize_use_llm=True,
    )
    history = [
        ChatTurn(role="user", content="What is an Elastic IP?"),
        ChatTurn(role="assistant", content="Static public IP [1]."),
    ]
    llm = _FakeLLM(rewritten="release Elastic IP address")
    rewritten = contextualize_query("How do I release it?", history, settings, llm)
    assert rewritten == "release Elastic IP address"
    assert llm.calls


def test_router_build_messages_includes_history():
    router = LLMRouter(Settings(llm_provider="extractive"))
    history = [
        ChatTurn(role="user", content="First question"),
        ChatTurn(role="assistant", content="First answer"),
    ]
    messages = router._build_messages("system", "current", history=history)
    assert messages[0]["role"] == "system"
    assert messages[1:3] == [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
    ]
    assert messages[-1] == {"role": "user", "content": "current"}
