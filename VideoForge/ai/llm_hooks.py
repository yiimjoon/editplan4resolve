"""Optional LLM hooks for transcript reflow/reorder."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List
from urllib import request
from urllib.error import HTTPError, URLError

from VideoForge.config.config_manager import Config

LEGACY_LLM_ENV_PATH = Path(__file__).resolve().parents[1] / "data" / "llm.env"


def _resolve_llm_env_path() -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        return (
            Path(appdata)
            / "Blackmagic Design"
            / "DaVinci Resolve"
            / "Support"
            / "VideoForge"
            / "llm.env"
        )
    return LEGACY_LLM_ENV_PATH


LLM_ENV_PATH = _resolve_llm_env_path()


def get_llm_provider() -> str | None:
    provider = os.environ.get("VIDEOFORGE_LLM_PROVIDER")
    if provider:
        env_provider = provider
    else:
        env = load_llm_env()
        env_provider = env.get("VIDEOFORGE_LLM_PROVIDER")
    try:
        cfg_provider = Config.get("llm_provider")
    except Exception:
        cfg_provider = None
    if cfg_provider is not None and str(cfg_provider).strip():
        return cfg_provider
    return env_provider


def is_llm_enabled() -> bool:
    provider = get_llm_provider()
    if not provider:
        return False
    provider = str(provider).strip().lower()
    if provider == "gemini":
        return bool(get_llm_api_key())
    return False


def get_llm_model() -> str:
    model = os.environ.get("VIDEOFORGE_LLM_MODEL")
    if model:
        return model
    env = load_llm_env()
    if env.get("VIDEOFORGE_LLM_MODEL"):
        return env["VIDEOFORGE_LLM_MODEL"]
    try:
        cfg_model = Config.get("llm_model")
        if cfg_model and str(cfg_model).strip():
            return cfg_model
        return "gemini-3-flash-preview"
    except Exception:
        return "gemini-3-flash-preview"


def get_llm_api_key() -> str | None:
    key = os.environ.get("VIDEOFORGE_GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    env = load_llm_env()
    key = env.get("VIDEOFORGE_GEMINI_API_KEY") or env.get("GEMINI_API_KEY")
    if key:
        return key
    try:
        cfg_key = Config.get("llm_api_key")
    except Exception:
        cfg_key = None
    if cfg_key and str(cfg_key).strip():
        return cfg_key
    return None


def get_llm_instructions() -> str:
    env = load_llm_env()
    if env.get("VIDEOFORGE_LLM_INSTRUCTIONS"):
        return env["VIDEOFORGE_LLM_INSTRUCTIONS"].replace("\\n", "\n").strip()
    try:
        return str(Config.get("llm_instructions", "")).strip()
    except Exception:
        return ""


def load_llm_env() -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not LLM_ENV_PATH.exists() and LEGACY_LLM_ENV_PATH.exists():
        try:
            LLM_ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
            LLM_ENV_PATH.write_text(
                LEGACY_LLM_ENV_PATH.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        except Exception:
            return env
    if not LLM_ENV_PATH.exists():
        return env
    try:
        for line in LLM_ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    except Exception:
        return {}
    return env


def write_llm_env(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    instructions: str | None = None,
) -> None:
    env = load_llm_env()
    if provider is not None:
        if provider:
            env["VIDEOFORGE_LLM_PROVIDER"] = provider
        else:
            env.pop("VIDEOFORGE_LLM_PROVIDER", None)
    if model is not None:
        if model:
            env["VIDEOFORGE_LLM_MODEL"] = model
        else:
            env.pop("VIDEOFORGE_LLM_MODEL", None)
    if api_key is not None:
        if api_key:
            env["VIDEOFORGE_GEMINI_API_KEY"] = api_key
        else:
            env.pop("VIDEOFORGE_GEMINI_API_KEY", None)
    if instructions is not None:
        if instructions:
            env["VIDEOFORGE_LLM_INSTRUCTIONS"] = instructions.replace("\n", "\\n")
        else:
            env.pop("VIDEOFORGE_LLM_INSTRUCTIONS", None)

    LLM_ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}={value}" for key, value in env.items()]
    LLM_ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _extract_json(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM response did not contain JSON payload.")
    payload = text[start : end + 1]
    return json.loads(payload)


def _call_gemini(prompt: str) -> str:
    api_key = get_llm_api_key()
    if not api_key:
        raise NotImplementedError(
            "Gemini API key missing. Set VIDEOFORGE_GEMINI_API_KEY or "
            "Config llm_api_key."
        )
    model = get_llm_model().strip()
    if model.startswith("models/"):
        model = model[len("models/"):]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3},
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"Gemini HTTP error: {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"Gemini network error: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Gemini request failed: {exc}") from exc

    try:
        return body["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as exc:
        raise RuntimeError(f"Gemini response parse failed: {exc}") from exc


def suggest_reflow_lines(
    sentences: List[Dict],
    max_chars: int,
) -> List[Dict]:
    provider = get_llm_provider()
    if provider != "gemini":
        raise NotImplementedError(
            "LLM provider not configured for reflow. Set VIDEOFORGE_LLM_PROVIDER=gemini."
        )

    lines = []
    for idx, sent in enumerate(sentences, start=1):
        text = str(sent.get("text", "")).strip()
        lines.append(f"{idx}. {text}")

    extra = get_llm_instructions()
    extra_block = f"\nUser instructions:\n{extra}\n" if extra else ""
    prompt = (
        "You are editing subtitle lines.\n"
        f"Rewrite each line so that it fits within {max_chars} characters.\n"
        "Keep meaning and language. Do not reorder lines.\n"
        "Return JSON: {\"lines\":[{\"index\":1,\"text\":\"...\"}, ...]}\n"
        + extra_block
        + "Lines:\n"
        + "\n".join(lines)
    )

    raw_text = _call_gemini(prompt)
    data = _extract_json(raw_text)
    updated = {int(item["index"]): str(item["text"]) for item in data.get("lines", [])}

    reflowed: List[Dict] = []
    for idx, sent in enumerate(sentences, start=1):
        new_text = updated.get(idx)
        if not new_text:
            reflowed.append(sent)
            continue
        clone = dict(sent)
        clone["text"] = new_text
        reflowed.append(clone)
    return reflowed


def suggest_reorder(
    sentences: List[Dict],
    profile: str = "hook_build_payoff",
    cut_strength: float = 0.2,
) -> Dict[str, Any]:
    provider = get_llm_provider()
    if provider != "gemini":
        raise NotImplementedError(
            "LLM provider not configured for reorder. Set VIDEOFORGE_LLM_PROVIDER=gemini."
        )

    lines = []
    for idx, sent in enumerate(sentences, start=1):
        text = str(sent.get("text", "")).strip()
        lines.append(f"{idx}. {text}")

    extra = get_llm_instructions()
    extra_block = f"\nUser instructions:\n{extra}\n" if extra else ""
    prompt = (
        "You are reordering transcript lines for short-form storytelling.\n"
        f"Profile: {profile}\n"
        "Goal: Hook -> Build -> Payoff. You may drop low-value filler lines.\n"
        f"Cut strength (0-1): {cut_strength}\n"
        "Return JSON only in this format:\n"
        "{\"order\":[1,2,3],\"drops\":[4,5],\"notes\":\"...\"}\n"
        + extra_block
        + "Lines:\n"
        + "\n".join(lines)
    )

    raw_text = _call_gemini(prompt)
    data = _extract_json(raw_text)

    def _to_int(value: Any) -> int | None:
        try:
            return int(value)
        except Exception:
            return None

    order = []
    for item in data.get("order", []):
        parsed = _to_int(item)
        if parsed is not None:
            order.append(parsed)

    drops = []
    for item in data.get("drops", []):
        parsed = _to_int(item)
        if parsed is not None:
            drops.append(parsed)
    return {"order": order, "drops": drops, "notes": data.get("notes", "")}
