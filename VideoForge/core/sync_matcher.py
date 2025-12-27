from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import numpy as np

try:
    from scipy.signal import correlate
except Exception:  # pragma: no cover - fallback when SciPy is unavailable
    correlate = None

from VideoForge.adapters.audio_sync import SilenceDetector
from VideoForge.config.config_manager import Config

logger = logging.getLogger(__name__)

SyncMode = Literal["same", "inverse"]


class SyncMatcher:
    """Compute sync offsets based on silence pattern matching."""

    def __init__(
        self,
        silence_detector: SilenceDetector | None = None,
        resolution: float | None = None,
    ) -> None:
        self.resolution = float(
            resolution
            if resolution is not None
            else Config.get("audio_sync_resolution", 0.1)
        )
        if self.resolution <= 0:
            self.resolution = 0.1
        threshold_db = float(Config.get("audio_sync_threshold_db", -40.0))
        min_silence = float(Config.get("audio_sync_min_silence", 0.3))
        self.max_offset_default = float(Config.get("audio_sync_max_offset", 30.0))
        self.min_confidence = float(Config.get("audio_sync_min_confidence", 0.5))
        self.detector = silence_detector or SilenceDetector(
            threshold_db=threshold_db,
            min_silence_duration=min_silence,
        )

    def find_sync_offset(
        self,
        reference_video: Path,
        target_video: Path,
        mode: SyncMode = "same",
        max_offset: float | None = None,
    ) -> Dict[str, Any]:
        """
        Compute sync offset between reference and target video.

        Returns:
            {
                "offset": float,
                "confidence": float,
                "ref_silences": List[Tuple[float, float]],
                "tgt_silences": List[Tuple[float, float]],
                "error": Optional[str]
            }
        """
        max_offset = float(max_offset) if max_offset is not None else self.max_offset_default
        if max_offset <= 0:
            max_offset = self.max_offset_default
        ref_silences = self.detector.detect_silence(Path(reference_video))
        tgt_silences = self.detector.detect_silence(Path(target_video))

        if not ref_silences or not tgt_silences:
            logger.warning("No silence detected in one or both videos.")
            return {
                "offset": 0.0,
                "confidence": 0.0,
                "ref_silences": ref_silences,
                "tgt_silences": tgt_silences,
                "error": "No silence patterns found",
            }

        logger.info(
            "Silence ranges: ref=%d target=%d",
            len(ref_silences),
            len(tgt_silences),
        )

        duration = max_offset * 2.0
        ref_pattern = self._silences_to_pattern(ref_silences, duration, self.resolution)
        tgt_pattern = self._silences_to_pattern(tgt_silences, duration, self.resolution)

        if not ref_pattern.any() or not tgt_pattern.any():
            logger.warning("Silence patterns empty after binning.")
            return {
                "offset": 0.0,
                "confidence": 0.0,
                "ref_silences": ref_silences,
                "tgt_silences": tgt_silences,
                "error": "No silence patterns found",
            }

        offset, confidence = self._correlate_patterns(
            ref_pattern,
            tgt_pattern,
            mode=mode,
            max_offset=max_offset,
        )

        logger.info(
            "Audio sync offset=%.3fs confidence=%.2f mode=%s",
            offset,
            confidence,
            mode,
        )
        if confidence < self.min_confidence:
            logger.warning("Low confidence: %.2f", confidence)

        return {
            "offset": offset,
            "confidence": confidence,
            "ref_silences": ref_silences,
            "tgt_silences": tgt_silences,
        }

    def sync_multiple(
        self,
        reference_video: Path,
        target_videos: List[Path],
        mode: SyncMode = "same",
    ) -> Dict[Path, Dict[str, Any]]:
        """Sync multiple target videos against a reference."""
        results: Dict[Path, Dict[str, Any]] = {}
        for target in target_videos:
            results[target] = self.find_sync_offset(
                reference_video=reference_video,
                target_video=target,
                mode=mode,
            )
        return results

    def _correlate_patterns(
        self,
        ref_pattern: np.ndarray,
        tgt_pattern: np.ndarray,
        mode: SyncMode,
        max_offset: float,
    ) -> Tuple[float, float]:
        if mode == "inverse":
            tgt_pattern = ~tgt_pattern
        ref = ref_pattern.astype(float)
        tgt = tgt_pattern.astype(float)
        if correlate is not None:
            correlation = correlate(ref, tgt, mode="full")
        else:
            correlation = np.correlate(ref, tgt, mode="full")
        center = len(ref_pattern) - 1
        max_frames = int(max_offset / self.resolution) if max_offset > 0 else center
        start = max(0, center - max_frames)
        end = min(len(correlation) - 1, center + max_frames)
        window = correlation[start : end + 1]
        if window.size == 0:
            return 0.0, 0.0
        best_rel = int(np.argmax(window))
        best_idx = start + best_rel
        max_corr = float(correlation[best_idx])
        offset_frames = best_idx - center
        offset_seconds = offset_frames * self.resolution
        max_possible = float(min(len(ref_pattern), len(tgt_pattern)))
        confidence = max_corr / max_possible if max_possible > 0 else 0.0
        return offset_seconds, confidence

    @staticmethod
    def _silences_to_pattern(
        silences: List[Tuple[float, float]],
        duration: float,
        resolution: float = 0.1,
    ) -> np.ndarray:
        n_frames = max(1, int(duration / resolution))
        pattern = np.zeros(n_frames, dtype=bool)
        for start, end in silences:
            start_idx = int(float(start) / resolution)
            end_idx = int(float(end) / resolution)
            if end_idx <= 0 or start_idx >= n_frames:
                continue
            start_idx = max(0, start_idx)
            end_idx = min(end_idx, n_frames)
            if end_idx > start_idx:
                pattern[start_idx:end_idx] = True
        return pattern
