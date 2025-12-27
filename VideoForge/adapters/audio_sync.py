from __future__ import annotations

import logging
import math
import subprocess
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _extract_pcm(video_path: Path, sample_rate: int) -> bytes:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=300)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Command timed out after 300s: {cmd[0]}") from exc
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))
    return proc.stdout


class SilenceDetector:
    """Extract silence ranges from a video's audio track."""

    def __init__(
        self,
        threshold_db: float = -40.0,
        min_silence_duration: float = 0.3,
        frame_duration: float = 0.05,
        sample_rate: int = 16000,
    ) -> None:
        self.threshold_db = float(threshold_db)
        self.min_silence_duration = max(0.0, float(min_silence_duration))
        self.frame_duration = max(0.01, float(frame_duration))
        self.sample_rate = max(1000, int(sample_rate))

    def detect_silence(self, video_path: Path) -> List[Tuple[float, float]]:
        """Return silence ranges [(start, end)] in seconds."""
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            samples, sr = self._extract_audio(path)
        except Exception as exc:
            logger.warning("Audio extraction failed: %s", exc)
            return []
        if samples.size == 0:
            return []
        frame_length = int(sr * self.frame_duration)
        if frame_length <= 0:
            return []
        rms = self._compute_rms(samples, frame_length)
        if rms.size == 0:
            return []
        db = 20.0 * np.log10(rms + 1e-10)
        is_silence = db < self.threshold_db
        total_duration = float(samples.size) / float(sr or 1)
        return self._get_continuous_ranges(
            is_silence, frame_length, sr, total_duration, self.min_silence_duration
        )

    def _extract_audio(self, video_path: Path) -> Tuple[np.ndarray, int]:
        pcm = _extract_pcm(video_path, self.sample_rate)
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        if samples.size:
            samples /= 32768.0
        return samples, self.sample_rate

    @staticmethod
    def _compute_rms(audio: np.ndarray, frame_length: int) -> np.ndarray:
        if frame_length <= 0 or audio.size == 0:
            return np.array([], dtype=np.float32)
        num_frames = int(math.ceil(audio.size / frame_length))
        if num_frames <= 0:
            return np.array([], dtype=np.float32)
        padded = audio
        pad = num_frames * frame_length - audio.size
        if pad > 0:
            padded = np.pad(audio, (0, pad))
        frames = padded.reshape(num_frames, frame_length)
        return np.sqrt(np.mean(frames * frames, axis=1))

    @staticmethod
    def _get_continuous_ranges(
        mask: np.ndarray,
        frame_length: int,
        sample_rate: int,
        total_duration: float,
        min_duration: float,
    ) -> List[Tuple[float, float]]:
        ranges: List[Tuple[float, float]] = []
        if mask.size == 0 or frame_length <= 0 or sample_rate <= 0:
            return ranges
        frame_duration = float(frame_length) / float(sample_rate)
        start_idx = None
        for idx, is_silence in enumerate(mask):
            if is_silence and start_idx is None:
                start_idx = idx
            elif not is_silence and start_idx is not None:
                start = start_idx * frame_duration
                end = idx * frame_duration
                if end - start >= min_duration:
                    ranges.append((start, min(end, total_duration)))
                start_idx = None
        if start_idx is not None:
            start = start_idx * frame_duration
            end = mask.size * frame_duration
            if end - start >= min_duration:
                ranges.append((start, min(end, total_duration)))
        return ranges
