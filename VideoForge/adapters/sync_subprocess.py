from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)

_LOGGED_SUBPROCESS_MODE = False
_LOGGED_WORKER_MISSING = False


def _log_worker_missing() -> None:
    global _LOGGED_WORKER_MISSING
    if _LOGGED_WORKER_MISSING:
        return
    logger.info("Audio sync worker not found; subprocess mode disabled.")
    _LOGGED_WORKER_MISSING = True


def _is_resolve_host() -> bool:
    exe = (sys.executable or "").lower()
    return exe.endswith("resolve.exe") or "davinci resolve" in exe


def _default_python_exe() -> Optional[Path]:
    try:
        import site

        for entry in site.getsitepackages():
            root = Path(str(entry))
            if root.name.lower() == "site-packages":
                root = root.parent.parent
            candidate = root / "python.exe"
            if candidate.exists():
                return candidate
    except Exception:
        return None
    return None


def _get_worker_python_exe() -> Optional[Path]:
    try:
        from VideoForge.config.config_manager import Config

        configured = str(Config.get("audio_sync_python_exe") or "").strip()
        if configured:
            path = Path(configured)
            if path.exists():
                return path
        fallback = str(Config.get("opencv_python_exe") or "").strip()
        if fallback:
            path = Path(fallback)
            if path.exists():
                return path
    except Exception:
        pass
    return _default_python_exe()


def worker_available() -> bool:
    return _get_worker_python_exe() is not None


def _normalize_mode(value: Any) -> str | bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("auto", ""):
            return "auto"
        if text in ("true", "1", "yes", "on"):
            return True
        if text in ("false", "0", "no", "off"):
            return False
    return "auto"


def should_use_subprocess(mode: str | None = None) -> bool:
    global _LOGGED_SUBPROCESS_MODE

    if os.environ.get("VIDEOFORGE_SYNC_WORKER"):
        return False

    try:
        from VideoForge.config.config_manager import Config

        value = _normalize_mode(Config.get("audio_sync_subprocess_mode"))
    except Exception:
        value = "auto"

    if value is False:
        return False

    if value is True:
        enabled = worker_available()
    else:
        if not _is_resolve_host():
            return False
        enabled = mode in ("voice", "content") and worker_available()

    if not enabled:
        _log_worker_missing()
        return False

    if not _LOGGED_SUBPROCESS_MODE:
        logger.info("Audio sync subprocess mode enabled.")
        _LOGGED_SUBPROCESS_MODE = True
    return True


def run_sync(
    reference_video: Path,
    target_video: Path,
    mode: str,
    max_offset: float | None = None,
    timeout_sec: float = 300.0,
) -> Dict[str, Any]:
    python_exe = _get_worker_python_exe()
    if not python_exe:
        return {
            "offset": 0.0,
            "confidence": 0.0,
            "error": "Audio sync worker python not found.",
        }

    args = [
        "--reference",
        str(reference_video),
        "--target",
        str(target_video),
        "--mode",
        str(mode),
    ]
    if max_offset is not None:
        args.extend(["--max-offset", str(max_offset)])

    cmd = [str(python_exe), "-m", "VideoForge.adapters.sync_worker", *args]
    env = os.environ.copy()
    env["VIDEOFORGE_SYNC_WORKER"] = "1"
    cwd = Path(__file__).resolve().parents[2]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            encoding="utf-8",
            errors="ignore",
            env=env,
            cwd=str(cwd),
        )
    except subprocess.TimeoutExpired:
        return {
            "offset": 0.0,
            "confidence": 0.0,
            "error": "Audio sync worker timed out.",
        }
    if proc.returncode != 0:
        return {
            "offset": 0.0,
            "confidence": 0.0,
            "error": proc.stderr.strip() or "Audio sync worker failed.",
        }
    payload = proc.stdout.strip()
    if not payload:
        return {
            "offset": 0.0,
            "confidence": 0.0,
            "error": "Audio sync worker returned no output.",
        }
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return {
            "offset": 0.0,
            "confidence": 0.0,
            "error": "Audio sync worker returned invalid JSON.",
        }
