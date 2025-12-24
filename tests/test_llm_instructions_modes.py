from __future__ import annotations

from pathlib import Path

import pytest

import VideoForge.ai.llm_hooks as llm_hooks


def test_get_llm_instructions_mode_specific_beats_generic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    env = {
        "VIDEOFORGE_LLM_INSTRUCTIONS": "GENERIC",
        "VIDEOFORGE_LLM_INSTRUCTIONS_LONGFORM": "LONG",
    }

    def fake_load_env() -> dict[str, str]:
        return dict(env)

    config_values = {
        "llm_instructions_shortform": "SHORT_CFG",
        "llm_instructions_longform": "LONG_CFG",
        "llm_instructions": "GENERIC_CFG",
    }

    monkeypatch.setattr(llm_hooks, "load_llm_env", fake_load_env)
    monkeypatch.setattr(
        llm_hooks.Config,
        "get",
        lambda key, default=None: config_values.get(key, default),
    )

    # shortform: mode-specific env missing, but longform env exists -> should NOT fall back to generic env
    assert llm_hooks.get_llm_instructions_for_mode("shortform") == "SHORT_CFG"
    # longform: env key exists -> use it
    assert llm_hooks.get_llm_instructions_for_mode("longform") == "LONG"


def test_get_llm_instructions_generic_env_used_when_no_mode_specific(monkeypatch: pytest.MonkeyPatch) -> None:
    env = {"VIDEOFORGE_LLM_INSTRUCTIONS": "GENERIC"}
    monkeypatch.setattr(llm_hooks, "load_llm_env", lambda: dict(env))
    monkeypatch.setattr(llm_hooks.Config, "get", lambda key, default=None: default)

    assert llm_hooks.get_llm_instructions_for_mode("shortform") == "GENERIC"
    assert llm_hooks.get_llm_instructions_for_mode("longform") == "GENERIC"


def test_write_llm_env_does_not_write_generic_for_mode_specific(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_path = tmp_path / "llm.env"
    monkeypatch.setattr(llm_hooks, "LLM_ENV_PATH", env_path)
    monkeypatch.setattr(llm_hooks, "LEGACY_LLM_ENV_PATH", tmp_path / "legacy.env")

    llm_hooks.write_llm_env(instructions="SHORT", instructions_mode="shortform")
    text = env_path.read_text(encoding="utf-8")
    assert "VIDEOFORGE_LLM_INSTRUCTIONS_SHORTFORM=SHORT" in text
    assert "VIDEOFORGE_LLM_INSTRUCTIONS=" not in text

