from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)

_LOGGED_SUBPROCESS_MODE = False
_LOGGED_WORKER_MISSING = False
_LOGGED_ZERO_METRICS: set[str] = set()
_LOGGED_BUILD_MARKER = False
_LOGGED_DEBUG_SAMPLE: set[str] = set()
_LOGGED_WORKER_ENV = False
_BUILD_MARKER = "p11-debug-03"


def _log_worker_missing() -> None:
    global _LOGGED_WORKER_MISSING
    if _LOGGED_WORKER_MISSING:
        return
    logger.info("OpenCV worker not found, using time-based sampling")
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

    cmd = [str(python_exe), "-m", "VideoForge.adapters.opencv_worker", *args]
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
        raise RuntimeError(proc.stderr.strip() or "opencv worker failed")
    payload = proc.stdout.strip()
    if not payload:
        return {}
    return json.loads(payload)


def _format_debug(debug: Any) -> str:
    if not isinstance(debug, dict):
        return ""
    try:
        return json.dumps(debug, ensure_ascii=False)
    except Exception:
        return str(debug)


def _metrics_all_zero(metrics: Dict[str, float]) -> bool:
    for key in ("sharpness", "motion", "stability", "face_score"):
        if float(metrics.get(key, 0.0) or 0.0) > 0.0:
            return False
    return True


def _log_build_marker() -> None:
    global _LOGGED_BUILD_MARKER
    if _LOGGED_BUILD_MARKER:
        return
    try:
        logger.info(
            "OpenCV subprocess build %s: %s",
            _BUILD_MARKER,
            str(Path(__file__).resolve()),
        )
    except Exception:
        logger.info("OpenCV subprocess build %s: <path unavailable>", _BUILD_MARKER)
    _LOGGED_BUILD_MARKER = True


def _get_worker_root() -> Optional[Path]:
    try:
        # ...\VideoForge\adapters\opencv_subprocess.py -> parents[2] == ...\Scripts\Comp\VideoForge
        return Path(__file__).resolve().parents[2]
    except Exception:
        return None


def _log_worker_env(python_exe: Path, worker_root: Optional[Path], env: Dict[str, str]) -> None:
    global _LOGGED_WORKER_ENV
    if _LOGGED_WORKER_ENV:
        return
    logger.info(
        "OpenCV worker env: python=%s root=%s PYTHONPATH=%s",
        str(python_exe),
        str(worker_root) if worker_root else "<none>",
        env.get("PYTHONPATH", ""),
    )
    _LOGGED_WORKER_ENV = True


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
            enabled = bool(value)

        if enabled:
            _log_build_marker()
            python_exe = _get_worker_python_exe()
            if not python_exe:
                _log_worker_missing()
                return False
            if not _LOGGED_SUBPROCESS_MODE:
                logger.info(
                    "OpenCV subprocess mode enabled (Resolve host). Worker python=%s",
                    str(python_exe),
                )
                _LOGGED_SUBPROCESS_MODE = True

        return enabled
    except Exception:
        _log_build_marker()
        python_exe = _get_worker_python_exe()
        if not python_exe:
            _log_worker_missing()
            return False
        if not _LOGGED_SUBPROCESS_MODE:
            logger.info(
                "OpenCV subprocess mode enabled (Resolve host). Worker python=%s",
                str(python_exe),
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


def run_angle_score(
    video_path: Path,
    timestamp_sec: float,
    face_detector: str = "opencv_dnn",
    timeout_sec: float = 120.0,
) -> Dict[str, float]:
    metrics, error, debug = run_angle_score_with_debug(
        video_path=video_path,
        timestamp_sec=timestamp_sec,
        face_detector=face_detector,
        timeout_sec=timeout_sec,
    )
    if error:
        logger.warning(
            "OpenCV angle_score worker error: %s (debug=%s)",
            error,
            _format_debug(debug),
        )
    if not error and _metrics_all_zero(metrics):
        key = str(video_path)
        if key not in _LOGGED_ZERO_METRICS:
            logger.warning(
                "OpenCV angle_score returned all-zero metrics for %s at %.2fs (debug=%s)",
                key,
                float(timestamp_sec),
                _format_debug(debug),
            )
            _LOGGED_ZERO_METRICS.add(key)
    return metrics


def run_angle_score_with_debug(
    video_path: Path,
    timestamp_sec: float,
    face_detector: str = "opencv_dnn",
    timeout_sec: float = 120.0,
) -> tuple[Dict[str, float], Optional[str], Dict[str, Any]]:
    result = _run_worker(
        [
            "angle_score",
            "--video",
            str(video_path),
            "--timestamp",
            str(float(timestamp_sec)),
            "--face-detector",
            str(face_detector),
        ],
        timeout_sec=timeout_sec,
    )
    error = result.get("error")
    debug = result.get("debug")
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        return {"sharpness": 0.0, "motion": 0.0, "stability": 0.0, "face_score": 0.0}, error, debug
    normalized = {
        "sharpness": float(metrics.get("sharpness", 0.0) or 0.0),
        "motion": float(metrics.get("motion", 0.0) or 0.0),
        "stability": float(metrics.get("stability", 0.0) or 0.0),
        "face_score": float(metrics.get("face_score", 0.0) or 0.0),
    }
    if debug:
        key = str(video_path)
        if key not in _LOGGED_DEBUG_SAMPLE:
            logger.info(
                "OpenCV angle_score debug sample for %s: %s",
                key,
                _format_debug(debug),
            )
            _LOGGED_DEBUG_SAMPLE.add(key)
    return normalized, error, debug
