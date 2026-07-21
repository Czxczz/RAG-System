"""LLM Router — hybrid local/cloud generation.

Routing modes:
  * auto       -> OpenAI if a key is configured, else Gemini if a key is
                  configured, else Ollama if reachable, else extractive.
  * openai     -> force OpenAI cloud (accuracy mode).
  * gemini     -> force Google Gemini cloud.
  * ollama     -> force local (privacy mode).
  * extractive -> no LLM at all; stitch an answer from the retrieved context.

Each generate() call returns (answer_text, provider_used) so callers can
surface which engine actually answered.
"""
from __future__ import annotations

import json
from collections.abc import Iterator

import httpx

from app.config import Settings

ChatMessage = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": str}


class LLMRouter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._openai_client = None

    # ── Provider availability ────────────────────────────────
    def _openai_available(self) -> bool:
        return bool(self.settings.openai_api_key)

    def _gemini_available(self) -> bool:
        return bool(self.settings.gemini_api_key)

    def _ollama_available(self) -> bool:
        try:
            resp = httpx.get(f"{self.settings.ollama_base_url}/api/tags", timeout=1.5)
            return resp.status_code == 200
        except Exception:
            return False

    def resolve_provider(self, mode: str) -> str:
        """Resolve a routing mode to a concrete provider (public)."""
        mode = (mode or "auto").lower()
        if mode in {"openai", "gemini", "ollama", "extractive"}:
            return mode

        # auto — honor Admin config ``llm_provider`` when that provider is usable.
        preferred = (self.settings.llm_provider or "auto").lower()
        if preferred == "openai" and self._openai_available():
            return "openai"
        if preferred == "gemini" and self._gemini_available():
            return "gemini"
        if preferred == "ollama":
            return "ollama" if self._ollama_available() else "extractive"
        if preferred == "extractive":
            return "extractive"

        if self._openai_available():
            return "openai"
        if self._gemini_available():
            return "gemini"
        if self._ollama_available():
            return "ollama"
        return "extractive"

    # Backwards-compatible private alias.
    def _resolve_provider(self, mode: str) -> str:
        return self.resolve_provider(mode)

    def has_llm(self, mode: str = "auto") -> bool:
        """True when a real (non-extractive) LLM provider is available."""
        return self.resolve_provider(mode) != "extractive"

    # ── Generation ───────────────────────────────────────────
    def generate(self, system_prompt: str, user_prompt: str, mode: str) -> tuple[str, str]:
        messages = self._build_messages(system_prompt, user_prompt)
        return self.generate_messages(messages, mode)

    def generate_with_history(
        self,
        system_prompt: str,
        user_prompt: str,
        mode: str,
        history: list,
    ) -> tuple[str, str]:
        messages = self._build_messages(system_prompt, user_prompt, history=history)
        return self.generate_messages(messages, mode)

    def generate_messages(self, messages: list[ChatMessage], mode: str) -> tuple[str, str]:
        primary = self._resolve_provider(mode)
        chain = [primary]
        if primary in {"openai", "gemini"} and self._ollama_available():
            chain.append("ollama")

        last_exc: Exception | None = None
        for provider in chain:
            try:
                if provider == "openai":
                    return self._generate_openai_messages(messages), "openai"
                if provider == "gemini":
                    return self._generate_gemini_messages(messages), "gemini"
                if provider == "ollama":
                    return self._generate_ollama_messages(messages), "ollama"
            except Exception as exc:
                last_exc = exc
                continue

        user_prompt = messages[-1]["content"] if messages else ""
        if primary == "extractive":
            return self._generate_extractive(user_prompt), "extractive"

        note = str(last_exc) if last_exc else "unknown error"
        return (
            self._generate_extractive(user_prompt, provider_failed=primary, error=note),
            "extractive",
        )

    @staticmethod
    def _build_messages(
        system_prompt: str,
        user_prompt: str,
        *,
        history: list | None = None,
    ) -> list[ChatMessage]:
        messages: list[ChatMessage] = [{"role": "system", "content": system_prompt}]
        if history:
            for turn in history:
                messages.append({"role": turn.role, "content": turn.content})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    # ── Streaming generation ─────────────────────────────────
    def generate_stream(
        self, system_prompt: str, user_prompt: str, mode: str
    ) -> Iterator[tuple[str, str]]:
        messages = self._build_messages(system_prompt, user_prompt)
        yield from self.generate_stream_messages(messages, mode)

    def generate_stream_with_history(
        self,
        system_prompt: str,
        user_prompt: str,
        mode: str,
        history: list,
    ) -> Iterator[tuple[str, str]]:
        messages = self._build_messages(system_prompt, user_prompt, history=history)
        yield from self.generate_stream_messages(messages, mode)

    def generate_stream_messages(
        self, messages: list[ChatMessage], mode: str
    ) -> Iterator[tuple[str, str]]:
        """Yield ``(delta_text, provider)`` tuples as the answer is produced.

        Mirrors :meth:`generate`'s routing + fallback cascade, with two
        important streaming caveats:

        * Fallback only happens *before* the first token is emitted. Once a
          provider has streamed any output we never switch providers mid-answer
          (that would produce a garbled, doubled response).
        * ``extractive`` is always the terminal fallback so the stream is never
          empty, even fully offline.
        """
        primary = self._resolve_provider(mode)
        chain = [primary]
        if primary in {"openai", "gemini"} and self._ollama_available():
            chain.append("ollama")
        if primary != "extractive":
            chain.append("extractive")

        user_prompt = messages[-1]["content"] if messages else ""
        last_exc: Exception | None = None
        for provider in chain:
            produced = False
            try:
                for delta in self._stream_provider_messages(provider, messages):
                    if not delta:
                        continue
                    produced = True
                    yield delta, provider
            except Exception as exc:
                last_exc = exc
                if produced:
                    return
                continue
            if produced:
                return

        text = self._generate_extractive(
            user_prompt,
            provider_failed=primary if primary != "extractive" else None,
            error=str(last_exc) if last_exc else None,
        )
        yield text, "extractive"

    def _stream_provider_messages(
        self, provider: str, messages: list[ChatMessage]
    ) -> Iterator[str]:
        if provider == "openai":
            yield from self._stream_openai_messages(messages)
        elif provider == "gemini":
            yield from self._stream_gemini_messages(messages)
        elif provider == "ollama":
            yield from self._stream_ollama_messages(messages)
        else:
            user_prompt = messages[-1]["content"] if messages else ""
            yield self._generate_extractive(user_prompt)

    def _stream_openai_messages(self, messages: list[ChatMessage]) -> Iterator[str]:
        if self._openai_client is None:
            from openai import OpenAI

            if not self.settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is not set.")
            self._openai_client = OpenAI(api_key=self.settings.openai_api_key)

        stream = self._openai_client.chat.completions.create(
            model=self.settings.openai_chat_model,
            temperature=0.1,
            stream=True,
            messages=messages,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def _stream_gemini_messages(self, messages: list[ChatMessage]) -> Iterator[str]:
        system_prompt, contents = self._gemini_from_messages(messages)
        if not self.settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")

        url = (
            f"{self.settings.gemini_base_url}/v1beta/models/"
            f"{self.settings.gemini_model}:streamGenerateContent?alt=sse"
        )
        with httpx.stream(
            "POST",
            url,
            headers={"x-goog-api-key": self.settings.gemini_api_key},
            json={
                "system_instruction": {"parts": [{"text": system_prompt}]},
                "contents": contents,
                "generationConfig": {"temperature": 0.1},
            },
            timeout=120.0,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                for cand in data.get("candidates", []):
                    for part in cand.get("content", {}).get("parts", []):
                        text = part.get("text")
                        if text:
                            yield text

    def _stream_ollama_messages(self, messages: list[ChatMessage]) -> Iterator[str]:
        with httpx.stream(
            "POST",
            f"{self.settings.ollama_base_url}/api/chat",
            json={
                "model": self.settings.ollama_model,
                "stream": True,
                "options": {
                    "temperature": 0.1,
                    "num_ctx": self.settings.ollama_num_ctx,
                },
                "messages": messages,
            },
            timeout=self.settings.ollama_timeout,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = data.get("message", {}).get("content")
                if content:
                    yield content
                if data.get("done"):
                    break

    def _generate_openai_messages(self, messages: list[ChatMessage]) -> str:
        if self._openai_client is None:
            from openai import OpenAI

            if not self.settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is not set.")
            self._openai_client = OpenAI(api_key=self.settings.openai_api_key)

        resp = self._openai_client.chat.completions.create(
            model=self.settings.openai_chat_model,
            temperature=0.1,
            messages=messages,
        )
        return (resp.choices[0].message.content or "").strip()

    def _generate_gemini_messages(self, messages: list[ChatMessage]) -> str:
        system_prompt, contents = self._gemini_from_messages(messages)
        if not self.settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")

        url = (
            f"{self.settings.gemini_base_url}/v1beta/models/"
            f"{self.settings.gemini_model}:generateContent"
        )
        resp = httpx.post(
            url,
            headers={"x-goog-api-key": self.settings.gemini_api_key},
            json={
                "system_instruction": {"parts": [{"text": system_prompt}]},
                "contents": contents,
                "generationConfig": {"temperature": 0.1},
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {data}")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()
        if not text:
            raise RuntimeError(f"Gemini returned an empty response: {data}")
        return text

    def _generate_ollama_messages(self, messages: list[ChatMessage]) -> str:
        resp = httpx.post(
            f"{self.settings.ollama_base_url}/api/chat",
            json={
                "model": self.settings.ollama_model,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_ctx": self.settings.ollama_num_ctx,
                },
                "messages": messages,
            },
            timeout=self.settings.ollama_timeout,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    @staticmethod
    def _gemini_from_messages(
        messages: list[ChatMessage],
    ) -> tuple[str, list[dict[str, object]]]:
        system_prompt = ""
        contents: list[dict[str, object]] = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
                continue
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        return system_prompt, contents

    def _generate_openai(self, system_prompt: str, user_prompt: str) -> str:
        return self._generate_openai_messages(
            self._build_messages(system_prompt, user_prompt)
        )

    def _generate_gemini(self, system_prompt: str, user_prompt: str) -> str:
        return self._generate_gemini_messages(
            self._build_messages(system_prompt, user_prompt)
        )

    def _generate_ollama(self, system_prompt: str, user_prompt: str) -> str:
        return self._generate_ollama_messages(
            self._build_messages(system_prompt, user_prompt)
        )

    def _generate_extractive(
        self,
        user_prompt: str,
        *,
        provider_failed: str | None = None,
        error: str | None = None,
    ) -> str:
        """Deterministic, no-LLM fallback.

        The orchestrator embeds the numbered context blocks inside user_prompt.
        We simply return them so the system stays useful (and fully offline)
        even with no LLM configured.
        """
        marker = "CONTEXT:"
        if marker in user_prompt:
            context = user_prompt.split(marker, 1)[1]
            context = context.split("QUESTION:", 1)[0].strip()
        else:
            context = user_prompt.strip()

        if provider_failed:
            intro = (
                f"The '{provider_failed}' provider was unavailable"
                + (f" ({error})" if error else "")
                + ", and the local fallback also failed or was not reachable. "
                "Here are the most relevant passages from your documents (cited below):\n\n"
            )
        else:
            intro = (
                "No LLM is configured, so here are the most relevant passages "
                "from your documents (cited below):\n\n"
            )
        return intro + context
