from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


class VideoAnalyzer:
    """OpenCV-based video analysis adapter."""

    @staticmethod
    def get_video_info(video_path: Path) -> Dict[str, float]:
        """
        Get basic video information.

        Returns:
            {
                "width": 1920,
                "height": 1080,
                "fps": 30.0,
                "frame_count": 1800,
                "duration": 60.0
            }
        """
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"width": 0, "height": 0, "fps": 0.0, "frame_count": 0, "duration": 0.0}

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0

        cap.release()
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
        }

    @staticmethod
    def extract_frame_at_time(
        video_path: Path,
        timestamp: float,
    ) -> Optional[object]:
        """
        Extract a frame at a given timestamp (seconds).

        Returns:
            Frame as BGR ndarray, or None if unavailable.
        """
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            cap.release()
            return None

        frame_idx = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        return frame if ret else None
