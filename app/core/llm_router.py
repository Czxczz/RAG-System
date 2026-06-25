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

import httpx

from app.config import Settings


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

    def _resolve_provider(self, mode: str) -> str:
        mode = (mode or "auto").lower()
        if mode in {"openai", "gemini", "ollama", "extractive"}:
            return mode
        # auto
        if self._openai_available():
            return "openai"
        if self._gemini_available():
            return "gemini"
        if self._ollama_available():
            return "ollama"
        return "extractive"

    # ── Generation ───────────────────────────────────────────
    def generate(self, system_prompt: str, user_prompt: str, mode: str) -> tuple[str, str]:
        provider = self._resolve_provider(mode)
        try:
            if provider == "openai":
                return self._generate_openai(system_prompt, user_prompt), "openai"
            if provider == "gemini":
                return self._generate_gemini(system_prompt, user_prompt), "gemini"
            if provider == "ollama":
                return self._generate_ollama(system_prompt, user_prompt), "ollama"
        except Exception as exc:  # graceful fallback keeps the API responsive
            return (
                self._generate_extractive(user_prompt)
                + f"\n\n(Note: '{provider}' provider failed: {exc}. "
                "Returned an extractive answer instead.)",
                "extractive",
            )
        return self._generate_extractive(user_prompt), "extractive"

    def _generate_openai(self, system_prompt: str, user_prompt: str) -> str:
        if self._openai_client is None:
            from openai import OpenAI

            if not self.settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is not set.")
            self._openai_client = OpenAI(api_key=self.settings.openai_api_key)

        resp = self._openai_client.chat.completions.create(
            model=self.settings.openai_chat_model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    def _generate_gemini(self, system_prompt: str, user_prompt: str) -> str:
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
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
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

    def _generate_ollama(self, system_prompt: str, user_prompt: str) -> str:
        resp = httpx.post(
            f"{self.settings.ollama_base_url}/api/chat",
            json={
                "model": self.settings.ollama_model,
                "stream": False,
                "options": {"temperature": 0.1},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    def _generate_extractive(self, user_prompt: str) -> str:
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
        return (
            "No LLM is configured, so here are the most relevant passages "
            "from your documents (cited below):\n\n" + context
        )
