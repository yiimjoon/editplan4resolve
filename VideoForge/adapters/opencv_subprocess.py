from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)

_LOGGED_SUBPROCESS_MODE = False


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

        configured = str(Config.get("opencv_python_exe") or "").strip()
        if configured:
            path = Path(configured)
            if path.exists():
                return path
    except Exception:
        pass
    return _default_python_exe()


def _run_worker(args: List[str], timeout_sec: float) -> Dict[str, Any]:
    python_exe = _get_worker_python_exe()
    if not python_exe:
        raise RuntimeError("opencv worker python executable not found")

    cmd = [str(python_exe), "-m", "VideoForge.adapters.opencv_worker", *args]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        encoding="utf-8",
        errors="ignore",
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "opencv worker failed")
    payload = proc.stdout.strip()
    if not payload:
        return {}
    return json.loads(payload)


def should_use_subprocess() -> bool:
    """
    Prefer subprocess mode inside Resolve host to avoid native-extension import hangs.
    """
    global _LOGGED_SUBPROCESS_MODE

    if not _is_resolve_host():
        try:
            from VideoForge.config.config_manager import Config

            value = Config.get("opencv_subprocess_mode")
            if value is None:
                return False
            return bool(value)
        except Exception:
            return False
    try:
        from VideoForge.config.config_manager import Config

        value = Config.get("opencv_subprocess_mode")
        if value is None:
            enabled = True
        else:
            enabled = True

        if enabled and not _LOGGED_SUBPROCESS_MODE:
            python_exe = _get_worker_python_exe()
            logger.info(
                "OpenCV subprocess mode enabled (Resolve host). Worker python=%s",
                str(python_exe) if python_exe else "(auto-detect failed)",
            )
            _LOGGED_SUBPROCESS_MODE = True

        return enabled
    except Exception:
        if not _LOGGED_SUBPROCESS_MODE:
            python_exe = _get_worker_python_exe()
            logger.info(
                "OpenCV subprocess mode enabled (Resolve host). Worker python=%s",
                str(python_exe) if python_exe else "(auto-detect failed)",
            )
            _LOGGED_SUBPROCESS_MODE = True
        return True


def run_scene_detect(
    video_path: Path,
    threshold: float,
    sample_rate: int,
    timeout_sec: float = 120.0,
) -> List[float]:
    result = _run_worker(
        [
            "scene_detect",
            "--video",
            str(video_path),
            "--threshold",
            str(threshold),
            "--sample-rate",
            str(int(sample_rate)),
        ],
        timeout_sec=timeout_sec,
    )
    scenes = result.get("scenes")
    if isinstance(scenes, list):
        return [float(x) for x in scenes if isinstance(x, (int, float, str))]
    return [0.0]


def run_quality_check(
    video_path: Path,
    sample_frames: int = 5,
    timeout_sec: float = 120.0,
) -> Dict[str, float]:
    result = _run_worker(
        [
            "quality_check",
            "--video",
            str(video_path),
            "--sample-frames",
            str(int(sample_frames)),
        ],
        timeout_sec=timeout_sec,
    )
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        return {"blur_score": 0.0, "brightness": 0.0, "noise_level": 0.0, "quality_score": 0.0}
    return {
        "blur_score": float(metrics.get("blur_score", 0.0) or 0.0),
        "brightness": float(metrics.get("brightness", 0.0) or 0.0),
        "noise_level": float(metrics.get("noise_level", 0.0) or 0.0),
        "quality_score": float(metrics.get("quality_score", 0.0) or 0.0),
    }
