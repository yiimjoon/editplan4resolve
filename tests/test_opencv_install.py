from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")


def test_opencv_basic() -> None:
    sample = (
        Path(__file__).resolve().parents[1]
        / "test_media"
        / "Aroll"
        / "sample1.mp4"
    )
    if not sample.exists():
        pytest.skip("sample video not available")

    cap = cv2.VideoCapture(str(sample))
    assert cap.isOpened(), "Failed to open video"

    ret, frame = cap.read()
    cap.release()

    assert ret, "Failed to read frame"
    assert frame is not None
    assert frame.ndim == 3 and frame.shape[2] == 3
