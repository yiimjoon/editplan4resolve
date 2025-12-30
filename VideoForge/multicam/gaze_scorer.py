"""Gaze (head-pose) scoring for multicam selection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

from VideoForge.config.config_manager import Config

logger = logging.getLogger(__name__)


class GazeScorer:
    """Estimate head yaw and convert to a gaze score."""

    def __init__(self) -> None:
        self.method = str(Config.get("multicam_gaze_method", "head_pose")).strip().lower()
        self.min_confidence = float(Config.get("multicam_gaze_min_confidence", 0.7))
        self.yaw_span = float(Config.get("multicam_gaze_yaw_span_deg", 35.0))
        self.last_debug: Optional[dict] = None
        self.last_error: Optional[str] = None

    def score_video_frame(
        self,
        video_path: Path,
        timestamp_sec: float,
        camera_yaw: Optional[float],
    ) -> float:
        if not self._enabled(camera_yaw):
            return 0.0

        try:
            from VideoForge.adapters.mediapipe_subprocess import (
                run_gaze_score_with_debug,
                should_use_subprocess,
            )

            if should_use_subprocess():
                yaw, confidence, error, debug = run_gaze_score_with_debug(
                    video_path=Path(video_path),
                    timestamp_sec=float(timestamp_sec),
                )
                self.last_error = error
                self.last_debug = debug
                if error:
                    return 0.0
                return self._calculate_gaze_score(float(yaw), float(confidence), float(camera_yaw))
        except Exception as exc:
            logger.debug("Mediapipe subprocess unavailable: %s", exc)

        try:
            from VideoForge.adapters.mediapipe_worker import estimate_yaw_from_video

            yaw, confidence, debug, error = estimate_yaw_from_video(
                Path(video_path), float(timestamp_sec)
            )
            self.last_debug = debug
            self.last_error = error
            if error:
                return 0.0
            return self._calculate_gaze_score(float(yaw), float(confidence), float(camera_yaw))
        except Exception as exc:
            self.last_error = str(exc)
            self.last_debug = None
            return 0.0

    def score_frame(self, frame, camera_yaw: Optional[float]) -> float:
        if not self._enabled(camera_yaw):
            return 0.0
        try:
            from VideoForge.adapters.mediapipe_worker import estimate_yaw_from_frame

            yaw, confidence, debug, error = estimate_yaw_from_frame(frame)
            self.last_debug = debug
            self.last_error = error
            if error:
                return 0.0
            return self._calculate_gaze_score(float(yaw), float(confidence), float(camera_yaw))
        except Exception as exc:
            self.last_error = str(exc)
            self.last_debug = None
            return 0.0

    def _enabled(self, camera_yaw: Optional[float]) -> bool:
        if camera_yaw is None:
            return False
        return self.method != "disabled"

    def _calculate_gaze_score(self, yaw_deg: float, confidence: float, camera_yaw: float) -> float:
        if confidence < self.min_confidence:
            return 0.0
        span = self.yaw_span if self.yaw_span > 0.1 else 35.0
        diff = abs(float(yaw_deg) - float(camera_yaw))
        score = max(0.0, 1.0 - (diff / span))
        return float(score * max(0.0, min(confidence, 1.0)))
