from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_LOGGED_SUBPROCESS_MODE = False
_LOGGED_WORKER_MISSING = False
_LOGGED_BUILD_MARKER = False
_LOGGED_DEBUG_SAMPLE: set[str] = set()
_LOGGED_WORKER_ENV = False
_BUILD_MARKER = "p12-beat-01"


def _log_worker_missing() -> None:
    global _LOGGED_WORKER_MISSING
    if _LOGGED_WORKER_MISSING:
        return
    logger.info("Beat worker not found, falling back to in-process detection")
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

        configured = str(Config.get("beat_python_exe") or "").strip()
        if not configured:
            configured = str(Config.get("opencv_python_exe") or "").strip()
        if configured:
            path = Path(configured)
            if path.exists():
                return path
    except Exception:
        pass
    return _default_python_exe()


def _get_worker_root() -> Optional[Path]:
    try:
        return Path(__file__).resolve().parents[2]
    except Exception:
        return None


def _log_build_marker() -> None:
    global _LOGGED_BUILD_MARKER
    if _LOGGED_BUILD_MARKER:
        return
    try:
        logger.info(
            "Beat subprocess build %s: %s",
            _BUILD_MARKER,
            str(Path(__file__).resolve()),
        )
    except Exception:
        logger.info("Beat subprocess build %s: <path unavailable>", _BUILD_MARKER)
    _LOGGED_BUILD_MARKER = True


def _log_worker_env(python_exe: Path, worker_root: Optional[Path], env: Dict[str, str]) -> None:
    global _LOGGED_WORKER_ENV
    if _LOGGED_WORKER_ENV:
        return
    logger.info(
        "Beat worker env: python=%s root=%s PYTHONPATH=%s",
        str(python_exe),
        str(worker_root) if worker_root else "<none>",
        env.get("PYTHONPATH", ""),
    )
    _LOGGED_WORKER_ENV = True


def should_use_subprocess() -> bool:
    global _LOGGED_SUBPROCESS_MODE
    if not _is_resolve_host():
        return False
    _log_build_marker()
    python_exe = _get_worker_python_exe()
    if not python_exe:
        _log_worker_missing()
        return False
    if not _LOGGED_SUBPROCESS_MODE:
        logger.info(
            "Beat subprocess mode enabled (Resolve host). Worker python=%s",
            str(python_exe),
        )
        _LOGGED_SUBPROCESS_MODE = True
    return True


def _run_worker(args: list[str], timeout_sec: float) -> Dict[str, Any]:
    python_exe = _get_worker_python_exe()
    if not python_exe:
        raise RuntimeError("beat worker python executable not found")

    env = os.environ.copy()
    worker_root = _get_worker_root()
    if worker_root:
        existing = env.get("PYTHONPATH", "")
        entries = [p for p in existing.split(os.pathsep) if p]
        root_str = str(worker_root)
        if root_str not in entries:
            entries.insert(0, root_str)
        env["PYTHONPATH"] = os.pathsep.join(entries)
    _log_worker_env(python_exe, worker_root, env)

    cmd = [str(python_exe), "-m", "VideoForge.adapters.beat_worker", *args]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        encoding="utf-8",
        errors="ignore",
        cwd=str(worker_root) if worker_root else None,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "beat worker failed")
    payload = proc.stdout.strip()
    if not payload:
        return {}
    return json.loads(payload)


def run_detect_beats_with_debug(
    audio_path: Path,
    mode: str,
    hop_length: int,
    onset_threshold: float,
    min_bpm: float,
    max_bpm: float,
    timeout_sec: float = 120.0,
) -> Tuple[Dict[str, Any], Optional[str], Dict[str, Any]]:
    result = _run_worker(
        [
            "detect_beats",
            "--audio",
            str(audio_path),
            "--mode",
            str(mode),
            "--hop-length",
            str(int(hop_length)),
            "--onset-threshold",
            str(float(onset_threshold)),
            "--min-bpm",
            str(float(min_bpm)),
            "--max-bpm",
            str(float(max_bpm)),
        ],
        timeout_sec=timeout_sec,
    )
    error = result.get("error")
    debug = result.get("debug")
    data = result.get("result")
    if debug:
        key = str(audio_path)
        if key not in _LOGGED_DEBUG_SAMPLE:
            logger.info(
                "Beat detect debug sample for %s: %s",
                key,
                json.dumps(debug, ensure_ascii=False),
            )
            _LOGGED_DEBUG_SAMPLE.add(key)
    if not isinstance(data, dict):
        return {}, error, debug if isinstance(debug, dict) else {}
    return data, error, debug if isinstance(debug, dict) else {}


def run_detect_beats(
    audio_path: Path,
    mode: str = "onset",
    hop_length: int = 512,
    onset_threshold: float = 0.5,
    min_bpm: float = 60.0,
    max_bpm: float = 180.0,
    timeout_sec: float = 120.0,
) -> Dict[str, Any]:
    data, error, debug = run_detect_beats_with_debug(
        audio_path=audio_path,
        mode=mode,
        hop_length=hop_length,
        onset_threshold=onset_threshold,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        timeout_sec=timeout_sec,
    )
    if error:
        logger.warning(
            "Beat detect worker error: %s (debug=%s)",
            error,
            json.dumps(debug, ensure_ascii=False),
        )
    return data

