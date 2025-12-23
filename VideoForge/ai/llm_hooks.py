"""Optional LLM hooks for transcript reflow (stub)."""

from __future__ import annotations

import os
from typing import Dict, List

from VideoForge.config.config_manager import Config


def get_llm_provider() -> str | None:
    provider = os.environ.get("VIDEOFORGE_LLM_PROVIDER")
    if provider:
        return provider
    try:
        return Config.get("llm_provider")
    except Exception:
        return None


def is_llm_enabled() -> bool:
    return bool(get_llm_provider())


def suggest_reflow_lines(
    sentences: List[Dict],
    max_chars: int,
) -> List[Dict]:
    """Stub for LLM-based transcript reflow.

    This is intentionally not implemented to avoid network calls in the plugin
    until a provider is configured.
    """
    provider = get_llm_provider()
    raise NotImplementedError(
        "LLM provider not configured. Set VIDEOFORGE_LLM_PROVIDER or "
        "implement suggest_reflow_lines() for provider=%s." % (provider or "unset")
    )
