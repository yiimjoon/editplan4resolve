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
_BUILD_MARKER = "p12-gaze-01"


def _log_worker_missing() -> None:
    global _LOGGED_WORKER_MISSING
    if _LOGGED_WORKER_MISSING:
        return
    logger.info("Mediapipe worker not found, gaze scoring disabled")
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

        configured = str(Config.get("mediapipe_python_exe") or "").strip()
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
            "Mediapipe subprocess build %s: %s",
            _BUILD_MARKER,
            str(Path(__file__).resolve()),
        )
    except Exception:
        logger.info("Mediapipe subprocess build %s: <path unavailable>", _BUILD_MARKER)
    _LOGGED_BUILD_MARKER = True


def _log_worker_env(python_exe: Path, worker_root: Optional[Path], env: Dict[str, str]) -> None:
    global _LOGGED_WORKER_ENV
    if _LOGGED_WORKER_ENV:
        return
    logger.info(
        "Mediapipe worker env: python=%s root=%s PYTHONPATH=%s",
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
            "Mediapipe subprocess mode enabled (Resolve host). Worker python=%s",
            str(python_exe),
        )
        _LOGGED_SUBPROCESS_MODE = True
    return True


def _run_worker(args: list[str], timeout_sec: float) -> Dict[str, Any]:
    python_exe = _get_worker_python_exe()
    if not python_exe:
        raise RuntimeError("mediapipe worker python executable not found")

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

    cmd = [str(python_exe), "-m", "VideoForge.adapters.mediapipe_worker", *args]
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
        raise RuntimeError(proc.stderr.strip() or "mediapipe worker failed")
    payload = proc.stdout.strip()
    if not payload:
        return {}
    return json.loads(payload)


def run_gaze_score_with_debug(
    video_path: Path,
    timestamp_sec: float,
    timeout_sec: float = 120.0,
) -> Tuple[float, float, Optional[str], Dict[str, Any]]:
    result = _run_worker(
        [
            "gaze_score",
            "--video",
            str(video_path),
            "--timestamp",
            str(float(timestamp_sec)),
        ],
        timeout_sec=timeout_sec,
    )
    error = result.get("error")
    debug = result.get("debug")
    yaw = float(result.get("yaw_deg", 0.0) or 0.0)
    confidence = float(result.get("confidence", 0.0) or 0.0)
    if debug:
        key = str(video_path)
        if key not in _LOGGED_DEBUG_SAMPLE:
            logger.info(
                "Mediapipe gaze_score debug sample for %s: %s",
                key,
                json.dumps(debug, ensure_ascii=False),
            )
            _LOGGED_DEBUG_SAMPLE.add(key)
    return yaw, confidence, error, debug


def run_gaze_score(
    video_path: Path,
    timestamp_sec: float,
    timeout_sec: float = 120.0,
) -> Tuple[float, float]:
    yaw, confidence, error, debug = run_gaze_score_with_debug(
        video_path=video_path,
        timestamp_sec=timestamp_sec,
        timeout_sec=timeout_sec,
    )
    if error:
        logger.warning(
            "Mediapipe gaze_score worker error: %s (debug=%s)",
            error,
            json.dumps(debug, ensure_ascii=False),
        )
    return yaw, confidence
