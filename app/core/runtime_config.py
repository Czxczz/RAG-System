"""Runtime configuration overrides persisted under ``data/runtime_settings.json``.

Buyers can change API keys and RAG thresholds from the admin UI without editing
``.env`` by hand. Overrides merge on top of environment / ``.env`` defaults.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Keys that may be edited via the admin config panel.
EDITABLE_KEYS: tuple[str, ...] = (
    "openai_api_key",
    "openai_chat_model",
    "gemini_api_key",
    "gemini_model",
    "ollama_base_url",
    "ollama_model",
    "llm_provider",
    "chunk_size",
    "chunk_overlap",
    "top_k",
    "min_score",
    "rerank_enabled",
    "retrieve_k",
    "hybrid_enabled",
    "bm25_top_k",
    "overview_demote_enabled",
    "mmr_enabled",
    "mmr_lambda",
    "mmr_dedup_threshold",
    "ocr_enabled",
    "ocr_language",
    "ocr_dpi",
    "ocr_min_chars_per_page",
    "retrieval_gate_enabled",
    "retrieval_gate_min_score",
    "answer_validation_enabled",
    "answer_validation_min_support",
    "prompt_injection_enabled",
    "prompt_injection_block",
    "max_upload_bytes",
)

SECRET_KEYS = frozenset({"openai_api_key", "gemini_api_key"})


def runtime_settings_path(data_dir: Path) -> Path:
    return data_dir / "runtime_settings.json"


def load_overrides(data_dir: Path) -> dict[str, Any]:
    path = runtime_settings_path(data_dir)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if k in EDITABLE_KEYS}


def save_overrides(data_dir: Path, updates: dict[str, Any]) -> dict[str, Any]:
    """Merge ``updates`` into the override file and return the full override map."""
    current = load_overrides(data_dir)
    for key, value in updates.items():
        if key not in EDITABLE_KEYS:
            continue
        # Empty string for secrets means "leave unchanged".
        if key in SECRET_KEYS and value in ("", None):
            continue
        # Ignore accidental paste of masked placeholder values from the UI.
        if key in SECRET_KEYS and isinstance(value, str) and "•" in value:
            continue
        current[key] = value
    path = runtime_settings_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return current


def describe_key_status(settings) -> str:
    """Short human-readable summary of which LLM keys are active."""
    parts: list[str] = []
    if settings.openai_api_key:
        parts.append("OpenAI key set")
    if settings.gemini_api_key:
        parts.append("Gemini key set")
    if not parts:
        return "No cloud API keys set (Ollama/extractive only)."
    return "; ".join(parts) + f". Default provider: {settings.llm_provider}."


def mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "••••••••"
    return value[:3] + "••••" + value[-2:]


def public_config_view(settings) -> dict[str, Any]:
    """Serialize editable settings for the admin UI (secrets masked)."""
    out: dict[str, Any] = {}
    for key in EDITABLE_KEYS:
        value = getattr(settings, key, None)
        if key in SECRET_KEYS:
            out[key] = mask_secret(str(value or ""))
            out[f"{key}_set"] = bool(value)
        else:
            out[key] = value
    out["auth_enabled"] = settings.auth_enabled
    return out
