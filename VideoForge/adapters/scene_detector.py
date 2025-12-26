from __future__ import annotations

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class SceneDetector:
    """Histogram-based scene change detector."""

    def __init__(self, threshold: float = 30.0) -> None:
        self.threshold = threshold

    def detect_scene_changes(
        self,
        video_path: Path,
        sample_rate: int = 1,
    ) -> List[float]:
        try:
            from VideoForge.adapters.opencv_subprocess import run_scene_detect, should_use_subprocess

            if should_use_subprocess():
                return run_scene_detect(
                    video_path=video_path,
                    threshold=float(self.threshold),
                    sample_rate=int(sample_rate),
                )
        except Exception:
            pass

        try:
            import cv2
        except Exception as exc:
            logger.warning("OpenCV unavailable for scene detection: %s", exc)
            return [0.0]

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning("Failed to open video for scene detection: %s", video_path)
            return [0.0]

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            cap.release()
            return [0.0]

        prev_hist = None
        scene_timestamps: List[float] = [0.0]
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if sample_rate > 1 and frame_idx % sample_rate != 0:
                frame_idx += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            if prev_hist is not None:
                diff = (
                    cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA) * 100.0
                )
                if diff > self.threshold:
                    timestamp = frame_idx / fps
                    scene_timestamps.append(timestamp)
                    logger.debug(
                        "Scene change at %.2fs (diff=%.1f)", timestamp, diff
                    )

            prev_hist = hist
            frame_idx += 1

        duration = frame_idx / fps if fps > 0 else 0.0
        cap.release()

        if not scene_timestamps or scene_timestamps[-1] < max(0.0, duration - 1.0):
            scene_timestamps.append(duration)

        logger.info(
            "Detected %d scene changes in %.1fs video",
            len(scene_timestamps),
            duration,
        )
        return scene_timestamps
