from pathlib import Path

import pytest

pytest.importorskip("cv2")

from VideoForge.adapters.video_analyzer import VideoAnalyzer


def test_video_analyzer_info() -> None:
    sample = (
        Path(__file__).resolve().parents[1]
        / "test_media"
        / "Aroll"
        / "sample1.mp4"
    )
    if not sample.exists():
        pytest.skip("sample video not available")

    info = VideoAnalyzer.get_video_info(sample)
    assert info["width"] > 0
    assert info["height"] > 0
    assert info["fps"] > 0
    assert info["frame_count"] > 0


def test_video_analyzer_frame() -> None:
    sample = (
        Path(__file__).resolve().parents[1]
        / "test_media"
        / "Aroll"
        / "sample1.mp4"
    )
    if not sample.exists():
        pytest.skip("sample video not available")

    frame = VideoAnalyzer.extract_frame_at_time(sample, 1.0)
    assert frame is not None
