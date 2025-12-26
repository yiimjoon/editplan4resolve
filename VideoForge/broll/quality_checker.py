from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


class VideoQualityChecker:
    """OpenCV-based video quality heuristics."""

    @staticmethod
    def calculate_blur_score(frame) -> float:
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    @staticmethod
    def calculate_brightness(frame) -> float:
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(gray.mean())

    @staticmethod
    def calculate_noise_level(frame) -> float:
        import cv2
        import numpy as np

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_strength = np.sqrt(sobelx**2 + sobely**2)
        return float(edge_strength.std())

    def check_video_quality(
        self,
        video_path: Path,
        sample_frames: int = 5,
    ) -> Dict[str, float]:
        try:
            from VideoForge.adapters.opencv_subprocess import run_quality_check, should_use_subprocess

            if should_use_subprocess():
                return run_quality_check(
                    video_path=video_path,
                    sample_frames=sample_frames,
                )
        except Exception:
            pass

        try:
            import cv2
            import numpy as np
        except Exception as exc:
            logger.warning("OpenCV unavailable for quality check: %s", exc)
            return {
                "blur_score": 0.0,
                "brightness": 0.0,
                "noise_level": 0.0,
                "quality_score": 0.0,
            }

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning("Failed to open video for quality check: %s", video_path)
            return {
                "blur_score": 0.0,
                "brightness": 0.0,
                "noise_level": 0.0,
                "quality_score": 0.0,
            }

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            return {
                "blur_score": 0.0,
                "brightness": 0.0,
                "noise_level": 0.0,
                "quality_score": 0.0,
            }

        sample_frames = max(1, sample_frames)
        start_idx = int(frame_count * 0.1)
        end_idx = max(start_idx + 1, int(frame_count * 0.9))
        frame_indices = np.linspace(start_idx, end_idx, sample_frames, dtype=int)

        blur_scores = []
        brightness_scores = []
        noise_scores = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            blur_scores.append(self.calculate_blur_score(frame))
            brightness_scores.append(self.calculate_brightness(frame))
            noise_scores.append(self.calculate_noise_level(frame))

        cap.release()

        if not blur_scores:
            return {
                "blur_score": 0.0,
                "brightness": 0.0,
                "noise_level": 0.0,
                "quality_score": 0.0,
            }

        avg_blur = float(np.mean(blur_scores))
        avg_brightness = float(np.mean(brightness_scores))
        avg_noise = float(np.mean(noise_scores))

        blur_quality = min(avg_blur / 150.0, 1.0)
        brightness_quality = 1.0 - min(abs(avg_brightness - 128.0) / 128.0, 1.0)
        noise_quality = max(1.0 - (avg_noise / 30.0), 0.0)

        quality_score = (
            blur_quality * 0.5
            + brightness_quality * 0.3
            + noise_quality * 0.2
        )

        return {
            "blur_score": avg_blur,
            "brightness": avg_brightness,
            "noise_level": avg_noise,
            "quality_score": quality_score,
        }
