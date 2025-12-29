from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import numpy as np

try:
    from scipy.signal import correlate
except Exception:  # pragma: no cover - fallback when SciPy is unavailable
    correlate = None

from VideoForge.adapters.audio_sync import _extract_pcm

logger = logging.getLogger(__name__)

ContentFeature = Literal["chroma", "onset"]


class ContentMatcher:
    """Match content patterns (melody/onset) using librosa features."""

    _librosa = None
    _load_error: str | None = None

    def __init__(
        self,
        feature: ContentFeature = "chroma",
        sample_rate: int = 22050,
        hop_length: int = 512,
        n_fft: int = 2048,
    ) -> None:
        self.feature = str(feature or "chroma").lower()
        if self.feature not in ("chroma", "onset"):
            self.feature = "chroma"
        self.sample_rate = max(1000, int(sample_rate))
        self.hop_length = max(64, int(hop_length))
        self.n_fft = max(self.hop_length, int(n_fft))

    @classmethod
    def is_available(cls) -> bool:
        if cls._load_error:
            return False
        try:
            return importlib.util.find_spec("librosa") is not None
        except Exception:
            return False

    @classmethod
    def _load_librosa(cls) -> bool:
        if cls._librosa is not None:
            return True
        if cls._load_error:
            return False
        try:
            import librosa

            cls._librosa = librosa
            return True
        except Exception as exc:
            cls._load_error = str(exc)
            logger.warning("Librosa unavailable: %s", exc)
            return False

    def match(
        self,
        reference_video: Path,
        target_video: Path,
        max_offset: float,
    ) -> Dict[str, Any]:
        if not self._load_librosa():
            return {
                "offset": 0.0,
                "confidence": 0.0,
                "error": "Content Pattern requires librosa.",
            }
        ref_features, frame_duration = self._extract_features(
            Path(reference_video), max_offset
        )
        tgt_features, _ = self._extract_features(Path(target_video), max_offset)
        if ref_features.size == 0 or tgt_features.size == 0:
            return {
                "offset": 0.0,
                "confidence": 0.0,
                "error": "No content patterns found",
            }
        if frame_duration <= 0:
            return {
                "offset": 0.0,
                "confidence": 0.0,
                "error": "Invalid frame duration",
            }
        ref_features = self._normalize_features(ref_features)
        tgt_features = self._normalize_features(tgt_features)
        offset, confidence = self._correlate_features(
            ref_features, tgt_features, frame_duration, max_offset
        )
        return {
            "offset": offset,
            "confidence": confidence,
            "feature": self.feature,
            "ref_frames": int(ref_features.shape[1]),
            "tgt_frames": int(tgt_features.shape[1]),
        }

    def _extract_features(
        self,
        video_path: Path,
        max_offset: float,
    ) -> Tuple[np.ndarray, float]:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            pcm = _extract_pcm(path, self.sample_rate)
        except Exception as exc:
            logger.warning("Audio extraction failed: %s", exc)
            return np.zeros((0, 0), dtype=np.float32), 0.0
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return np.zeros((0, 0), dtype=np.float32), 0.0
        samples /= 32768.0
        if max_offset > 0:
            max_samples = int(max_offset * 2.0 * self.sample_rate)
            if max_samples > 0:
                samples = samples[:max_samples]
        librosa = self._librosa
        if self.feature == "onset":
            onset = librosa.onset.onset_strength(
                y=samples, sr=self.sample_rate, hop_length=self.hop_length
            )
            features = onset.reshape(1, -1).astype(np.float32)
        else:
            chroma = librosa.feature.chroma_stft(
                y=samples,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )
            features = chroma.astype(np.float32)
        frame_duration = float(self.hop_length) / float(self.sample_rate)
        return features, frame_duration

    @staticmethod
    def _normalize_features(features: np.ndarray) -> np.ndarray:
        if features.size == 0:
            return features
        norm = np.linalg.norm(features, axis=0)
        norm[norm == 0] = 1.0
        return features / norm

    @staticmethod
    def _correlate_features(
        ref_features: np.ndarray,
        tgt_features: np.ndarray,
        frame_duration: float,
        max_offset: float,
    ) -> Tuple[float, float]:
        dims = min(ref_features.shape[0], tgt_features.shape[0])
        if dims <= 0:
            return 0.0, 0.0
        ref_frames = ref_features.shape[1]
        tgt_frames = tgt_features.shape[1]
        if ref_frames == 0 or tgt_frames == 0:
            return 0.0, 0.0
        correlation = None
        for idx in range(dims):
            ref = ref_features[idx]
            tgt = tgt_features[idx]
            if correlate is not None:
                corr = correlate(ref, tgt, mode="full")
            else:
                corr = np.correlate(ref, tgt, mode="full")
            correlation = corr if correlation is None else correlation + corr
        if correlation is None:
            return 0.0, 0.0
        center = ref_frames - 1
        max_frames = int(max_offset / frame_duration) if max_offset > 0 else center
        start = max(0, center - max_frames)
        end = min(len(correlation) - 1, center + max_frames)
        window = correlation[start : end + 1]
        if window.size == 0:
            return 0.0, 0.0
        best_rel = int(np.argmax(window))
        best_idx = start + best_rel
        max_corr = float(correlation[best_idx])
        offset_frames = best_idx - center
        offset_seconds = offset_frames * frame_duration
        max_possible = float(min(ref_frames, tgt_frames))
        confidence = max_corr / max_possible if max_possible > 0 else 0.0
        return offset_seconds, confidence
