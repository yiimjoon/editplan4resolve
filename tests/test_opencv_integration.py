from pathlib import Path

import pytest

pytest.importorskip("cv2")

from VideoForge.adapters.scene_detector import SceneDetector
from VideoForge.broll.quality_checker import VideoQualityChecker


def _sample_video() -> Path | None:
    sample = (
        Path(__file__).resolve().parents[1]
        / "test_media"
        / "Aroll"
        / "sample1.mp4"
    )
    return sample if sample.exists() else None


def test_scene_detection() -> None:
    sample = _sample_video()
    if not sample:
        pytest.skip("sample video not available")

    detector = SceneDetector(threshold=30.0)
    scenes = detector.detect_scene_changes(sample, sample_rate=5)
    assert scenes
    assert scenes[0] == 0.0


def test_quality_check() -> None:
    sample = _sample_video()
    if not sample:
        pytest.skip("sample video not available")

    checker = VideoQualityChecker()
    quality = checker.check_video_quality(sample, sample_frames=3)
    assert 0.0 <= quality["quality_score"] <= 1.0
