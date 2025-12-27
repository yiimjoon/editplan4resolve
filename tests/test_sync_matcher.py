from pathlib import Path

from VideoForge.core.sync_matcher import SyncMatcher


class DummyDetector:
    def __init__(self, mapping):
        self.mapping = mapping

    def detect_silence(self, video_path: Path):
        return self.mapping[str(video_path)]


def test_same_pattern_sync():
    """Case 1: multi-camera same-pattern sync."""
    ref = Path("camera1.mp4")
    tgt = Path("camera2.mp4")
    mapping = {
        str(ref): [(0.0, 4.0)],
        str(tgt): [(2.0, 6.0)],
    }
    matcher = SyncMatcher(silence_detector=DummyDetector(mapping), resolution=1.0)
    result = matcher.find_sync_offset(ref, tgt, mode="same", max_offset=3.0)
    assert "offset" in result
    assert "confidence" in result
    assert result["confidence"] > 0.5
    assert abs(result["offset"] - (-2.0)) <= 0.5


def test_inverse_pattern_sync():
    """Case 2: inverse-pattern sync."""
    ref = Path("aroll_face.mp4")
    tgt = Path("broll_screen.mp4")
    mapping = {
        str(ref): [(0.0, 4.0)],
        str(tgt): [(4.0, 6.0)],
    }
    matcher = SyncMatcher(silence_detector=DummyDetector(mapping), resolution=1.0)
    result = matcher.find_sync_offset(ref, tgt, mode="inverse", max_offset=3.0)
    assert result["confidence"] > 0.5
    assert abs(result["offset"]) <= 0.5


def test_multiple_sync():
    """Case 3: multi-target sync."""
    ref = Path("camera1.mp4")
    tgt_a = Path("camera2.mp4")
    tgt_b = Path("camera3.mp4")
    mapping = {
        str(ref): [(0.0, 4.0)],
        str(tgt_a): [(2.0, 6.0)],
        str(tgt_b): [(1.0, 5.0)],
    }
    matcher = SyncMatcher(silence_detector=DummyDetector(mapping), resolution=1.0)
    matcher.max_offset_default = 3.0
    results = matcher.sync_multiple(ref, [tgt_a, tgt_b], mode="same")
    assert len(results) == 2
    result_a = results[tgt_a]
    result_b = results[tgt_b]
    assert result_a["confidence"] > 0.5
    assert result_b["confidence"] > 0.5
    assert abs(result_a["offset"] - (-2.0)) <= 0.5
    assert abs(result_b["offset"] - (-1.0)) <= 0.5
