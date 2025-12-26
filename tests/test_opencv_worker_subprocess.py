import json
import subprocess
from pathlib import Path

import pytest


def _sample_video() -> Path | None:
    sample = (
        Path(__file__).resolve().parents[1]
        / "test_media"
        / "Aroll"
        / "sample1.mp4"
    )
    return sample if sample.exists() else None


def _find_worker_python() -> Path | None:
    candidates = [
        Path(r"C:\Users\82102\AppData\Local\Programs\Python\Python312\python.exe"),
        Path(r"C:\Python312\python.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _worker_has_cv2(python_exe: Path) -> bool:
    proc = subprocess.run(
        [str(python_exe), "-c", "import cv2; import numpy as np; print(cv2.__version__)"],
        capture_output=True,
        text=True,
        timeout=60,
        encoding="utf-8",
        errors="ignore",
    )
    return proc.returncode == 0 and bool(proc.stdout.strip())


def test_opencv_worker_scene_detect_subprocess() -> None:
    sample = _sample_video()
    if not sample:
        pytest.skip("sample video not available")

    python_exe = _find_worker_python()
    if not python_exe or not _worker_has_cv2(python_exe):
        pytest.skip("opencv worker python not available")

    proc = subprocess.run(
        [
            str(python_exe),
            "-m",
            "VideoForge.adapters.opencv_worker",
            "scene_detect",
            "--video",
            str(sample),
            "--threshold",
            "30",
            "--sample-rate",
            "10",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        encoding="utf-8",
        errors="ignore",
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip())
    assert isinstance(payload.get("scenes"), list)


def test_opencv_worker_quality_check_subprocess() -> None:
    sample = _sample_video()
    if not sample:
        pytest.skip("sample video not available")

    python_exe = _find_worker_python()
    if not python_exe or not _worker_has_cv2(python_exe):
        pytest.skip("opencv worker python not available")

    proc = subprocess.run(
        [
            str(python_exe),
            "-m",
            "VideoForge.adapters.opencv_worker",
            "quality_check",
            "--video",
            str(sample),
            "--sample-frames",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        encoding="utf-8",
        errors="ignore",
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip())
    metrics = payload.get("metrics")
    assert isinstance(metrics, dict)
    assert "quality_score" in metrics
