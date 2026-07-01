"""In-memory conversation history for multi-turn chat.

Stores prior user/assistant turns per ``conversation_id`` and helps follow-up
questions work in a RAG setting:

  * **Retrieval** — contextualize short or referential follow-ups into a
    standalone search query so FAISS still finds the right chunks.
  * **Generation** — pass recent turns to the LLM so pronouns like "it" /
    "those" resolve against the chat, while the current turn still includes the
    fresh CONTEXT block with new citations.

Storage is process-local (like the FAISS index). A future Web UI can keep the
``conversation_id`` in browser state and send it on every ``/chat`` request.
"""
from __future__ import annotations

import datetime as dt
import re
import threading
import uuid
from dataclasses import dataclass, field
from typing import Literal

from app.config import Settings
from app.core.llm_router import LLMRouter

Role = Literal["user", "assistant"]

_REFERENTIAL = re.compile(
    r"\b(?:it|its|they|them|their|that|those|this|these|also|more|"
    r"above|earlier|previous|same|another|else|instead)\b",
    re.IGNORECASE,
)

_CONTEXTUALIZE_SYSTEM = (
    "You rewrite a follow-up question into a standalone search query for a "
    "document retrieval system. Use the chat history only to resolve pronouns "
    "and missing context. Output ONLY the rewritten query — no quotes, "
    "numbering, or explanation."
)


@dataclass
class ChatTurn:
    role: Role
    content: str


@dataclass
class Conversation:
    id: str
    turns: list[ChatTurn] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


class ConversationStore:
    """Thread-safe, in-process store keyed by conversation id."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conversations: dict[str, Conversation] = {}

    def get_or_create(self, conversation_id: str | None = None) -> tuple[str, Conversation]:
        with self._lock:
            if conversation_id and conversation_id in self._conversations:
                return conversation_id, self._conversations[conversation_id]
            new_id = conversation_id or uuid.uuid4().hex
            now = _now_iso()
            conv = Conversation(id=new_id, created_at=now, updated_at=now)
            self._conversations[new_id] = conv
            return new_id, conv

    def get(self, conversation_id: str) -> Conversation | None:
        with self._lock:
            return self._conversations.get(conversation_id)

    def recent_turns(self, conversation_id: str, max_turns: int) -> list[ChatTurn]:
        """Return up to ``max_turns`` most recent messages (user + assistant)."""
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if not conv or max_turns <= 0:
                return []
            return list(conv.turns[-max_turns:])

    def append_turn(self, conversation_id: str, role: Role, content: str) -> None:
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if conv is None:
                return
            conv.turns.append(ChatTurn(role=role, content=content))
            conv.updated_at = _now_iso()

    def append_exchange(
        self, conversation_id: str, user_message: str, assistant_message: str, *, max_turns: int
    ) -> None:
        """Record a user/assistant pair and trim to the configured window."""
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if conv is None:
                return
            conv.turns.append(ChatTurn(role="user", content=user_message))
            conv.turns.append(ChatTurn(role="assistant", content=assistant_message))
            if max_turns > 0 and len(conv.turns) > max_turns:
                conv.turns = conv.turns[-max_turns:]
            conv.updated_at = _now_iso()

    def delete(self, conversation_id: str) -> bool:
        with self._lock:
            if conversation_id in self._conversations:
                del self._conversations[conversation_id]
                return True
            return False


def needs_contextualization(query: str, history: list[ChatTurn]) -> bool:
    """True when the query likely depends on prior chat to retrieve well."""
    if not history:
        return False
    if _REFERENTIAL.search(query):
        return True
    # Short follow-ups ("and spot?", "pricing?") benefit from history.
    return len(query.split()) < 6


def contextualize_query(
    query: str,
    history: list[ChatTurn],
    settings: Settings,
    llm: LLMRouter,
) -> str:
    """Rewrite a follow-up into a standalone retrieval query."""
    if not history or not settings.chat_memory_enabled:
        return query
    if not settings.chat_memory_contextualize:
        return query
    if not needs_contextualization(query, history):
        return query

    if settings.chat_memory_contextualize_use_llm and llm.has_llm("auto"):
        try:
            return _llm_contextualize(query, history, llm)
        except Exception:
            pass
    return _heuristic_contextualize(query, history)


def format_history_block(history: list[ChatTurn]) -> str:
    """Compact text block for heuristic contextualization."""
    lines: list[str] = []
    for turn in history[-6:]:
        label = "User" if turn.role == "user" else "Assistant"
        lines.append(f"{label}: {turn.content}")
    return "\n".join(lines)


def _llm_contextualize(query: str, history: list[ChatTurn], llm: LLMRouter) -> str:
    history_text = format_history_block(history)
    user_prompt = (
        f"CHAT HISTORY:\n{history_text}\n\n"
        f"FOLLOW-UP QUESTION: {query}\n\n"
        "Standalone search query:"
    )
    text, _ = llm.generate(_CONTEXTUALIZE_SYSTEM, user_prompt, mode="auto")
    rewritten = " ".join(text.strip().splitlines()[0].split())
    return rewritten or query


def _heuristic_contextualize(query: str, history: list[ChatTurn]) -> str:
    last_user = ""
    last_assistant = ""
    for turn in reversed(history):
        if turn.role == "assistant" and not last_assistant:
            last_assistant = turn.content[:240]
        elif turn.role == "user" and not last_user:
            last_user = turn.content
        if last_user and last_assistant:
            break
    if last_user:
        return f"{last_user} — follow-up: {query}"
    return query


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()
