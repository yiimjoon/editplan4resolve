"""Audio beat/onset detection utilities."""

from __future__ import annotations

import json
import logging
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from VideoForge.config.config_manager import Config

logger = logging.getLogger(__name__)


class BeatDetector:
    """Detect rhythmic events from an audio file."""

    def __init__(
        self,
        mode: Optional[str] = None,
        hop_length: Optional[int] = None,
        onset_threshold: Optional[float] = None,
        min_bpm: Optional[float] = None,
        max_bpm: Optional[float] = None,
    ) -> None:
        self.mode = str(mode or Config.get("beat_detector_mode", "onset")).strip().lower()
        self.hop_length = int(hop_length or Config.get("beat_detector_hop_length", 512))
        self.onset_threshold = float(
            onset_threshold if onset_threshold is not None else Config.get("beat_detector_onset_threshold", 0.5)
        )
        self.min_bpm = float(min_bpm or Config.get("beat_detector_min_bpm", 60))
        self.max_bpm = float(max_bpm or Config.get("beat_detector_max_bpm", 180))
        self.downbeat_weight = float(Config.get("beat_detector_downbeat_weight", 0.3))
        self.last_debug: Optional[Dict[str, Any]] = None
        self.last_error: Optional[str] = None

    def detect(self, audio_path: Path, mode: Optional[str] = None) -> Dict[str, Any]:
        self.last_debug = {}
        self.last_error = None
        mode = str(mode or self.mode or "onset").strip().lower()
        audio_path = Path(audio_path)

        try:
            beats, debug = self._detect_with_librosa(audio_path, mode)
            self.last_debug = debug
            return beats
        except Exception as exc:
            self.last_debug = {"backend": "fallback", "error": str(exc)}
            logger.debug("Beat detector: librosa unavailable or failed: %s", exc)

        try:
            beats, debug = self._detect_fallback(audio_path, mode)
            self.last_debug = debug
            return beats
        except Exception as exc:
            self.last_error = str(exc)
            self.last_debug = {"backend": "fallback", "error": str(exc)}
            return self._empty_result()

    def _detect_with_librosa(
        self, audio_path: Path, mode: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        import librosa  # type: ignore

        y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
        duration = float(len(y) / sr) if sr else 0.0
        debug: Dict[str, Any] = {
            "backend": "librosa",
            "sr": sr,
            "duration": duration,
            "hop_length": self.hop_length,
        }

        onset_frames = librosa.onset.onset_detect(
            y=y,
            sr=sr,
            hop_length=self.hop_length,
            backtrack=True,
        )
        onsets = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length)

        tempo, beat_frames = librosa.beat.beat_track(
            y=y,
            sr=sr,
            hop_length=self.hop_length,
            start_bpm=max(self.min_bpm, 60.0),
        )
        if tempo < self.min_bpm:
            tempo = self.min_bpm
        if tempo > self.max_bpm:
            tempo = self.max_bpm
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=self.hop_length)

        downbeats = self._derive_downbeats(beat_times)
        segments = self._segment_from_beats(beat_times, duration)
        return (
            {
                "bpm": float(tempo or 0.0),
                "onsets": [float(x) for x in onsets],
                "beats": [float(x) for x in beat_times],
                "downbeats": [float(x) for x in downbeats],
                "segments": segments,
                "duration": duration,
                "mode": mode,
            },
            debug,
        )

    def _detect_fallback(
        self, audio_path: Path, mode: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        import numpy as np  # type: ignore

        audio_path = self._ensure_wav(audio_path)
        y, sr = self._read_wave(audio_path)
        if y is None or sr <= 0:
            raise RuntimeError("Failed to read audio for beat detection.")
        duration = float(len(y) / sr)
        debug: Dict[str, Any] = {
            "backend": "rms",
            "sr": sr,
            "duration": duration,
            "hop_length": self.hop_length,
        }

        hop = max(1, int(self.hop_length))
        frame_count = max(1, int(math.ceil(len(y) / hop)))
        rms = []
        for idx in range(frame_count):
            start = idx * hop
            end = min(len(y), start + hop)
            frame = y[start:end]
            if frame.size == 0:
                rms.append(0.0)
            else:
                rms.append(float(np.sqrt(np.mean(frame**2))))
        rms_arr = np.array(rms, dtype=np.float32)
        onset_env = np.maximum(0.0, np.diff(rms_arr, prepend=rms_arr[0]))
        threshold = float(np.max(onset_env) * float(self.onset_threshold or 0.0))
        if threshold <= 0:
            threshold = float(np.percentile(onset_env, 75)) if onset_env.size else 0.0
        onset_frames = np.where(onset_env >= threshold)[0] if onset_env.size else np.array([])
        onsets = (onset_frames * hop) / float(sr)

        bpm = self._estimate_bpm_from_onsets(onsets)
        if mode == "onset":
            beat_times = list(onsets)
        else:
            beat_times = self._grid_from_bpm(bpm, duration, onsets)
        downbeats = self._derive_downbeats(beat_times)
        segments = self._segment_from_beats(beat_times, duration)
        return (
            {
                "bpm": float(bpm),
                "onsets": [float(x) for x in onsets],
                "beats": [float(x) for x in beat_times],
                "downbeats": [float(x) for x in downbeats],
                "segments": segments,
                "duration": duration,
                "mode": mode,
            },
            debug,
        )

    def _estimate_bpm_from_onsets(self, onsets: Any) -> float:
        try:
            import numpy as np  # type: ignore

            if onsets is None or len(onsets) < 2:
                return float(self.min_bpm)
            intervals = np.diff(onsets)
            intervals = intervals[intervals > 0]
            if intervals.size == 0:
                return float(self.min_bpm)
            median_interval = float(np.median(intervals))
            if median_interval <= 0:
                return float(self.min_bpm)
            bpm = 60.0 / median_interval
            return float(min(max(bpm, self.min_bpm), self.max_bpm))
        except Exception:
            return float(self.min_bpm)

    def _grid_from_bpm(self, bpm: float, duration: float, onsets: Any) -> List[float]:
        if bpm <= 0 or duration <= 0:
            return [0.0, duration] if duration > 0 else [0.0]
        period = 60.0 / bpm
        start = 0.0
        if onsets is not None and len(onsets):
            try:
                start = float(onsets[0])
            except Exception:
                start = 0.0
        beats: List[float] = []
        t = max(0.0, start)
        while t < duration:
            beats.append(float(t))
            t += period
        if not beats or beats[-1] < duration:
            beats.append(float(duration))
        return beats

    @staticmethod
    def _derive_downbeats(beats: List[float]) -> List[float]:
        if not beats:
            return []
        return [float(beats[i]) for i in range(0, len(beats), 4)]

    def _segment_from_beats(self, beats: List[float], duration: float) -> List[Dict[str, Any]]:
        if duration <= 0:
            return []
        if not beats:
            return [{"start": 0.0, "end": float(duration), "type": "segment"}]
        segments: List[Dict[str, Any]] = []
        beat_group = 16
        idx = 0
        while idx < len(beats) - 1:
            start = beats[idx]
            end_idx = min(idx + beat_group, len(beats) - 1)
            end = beats[end_idx]
            seg_type = "segment"
            if start <= duration * 0.1:
                seg_type = "intro"
            elif end >= duration * 0.9:
                seg_type = "outro"
            segments.append({"start": float(start), "end": float(end), "type": seg_type})
            idx = end_idx
        if segments and segments[-1]["end"] < duration:
            segments[-1]["end"] = float(duration)
        return segments

    def _ensure_wav(self, audio_path: Path) -> Path:
        if audio_path.suffix.lower() == ".wav":
            return audio_path
        tmp = Path(tempfile.gettempdir()) / f"videoforge_beat_{audio_path.stem}.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "22050",
            str(tmp),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "ffmpeg conversion failed")
        return tmp

    def _read_wave(self, audio_path: Path) -> Tuple[Optional["numpy.ndarray"], int]:
        try:
            import soundfile as sf  # type: ignore

            data, sr = sf.read(str(audio_path), dtype="float32")
            if hasattr(data, "ndim") and data.ndim > 1:
                data = data.mean(axis=1)
            return data, int(sr)
        except Exception:
            pass

        try:
            import wave
            import numpy as np  # type: ignore

            with wave.open(str(audio_path), "rb") as handle:
                sr = handle.getframerate()
                frames = handle.readframes(handle.getnframes())
                if not frames:
                    return None, int(sr)
                dtype = np.int16
                data = np.frombuffer(frames, dtype=dtype).astype("float32") / 32768.0
                return data, int(sr)
        except Exception as exc:
            raise RuntimeError(f"wave read failed: {exc}") from exc

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "bpm": 0.0,
            "onsets": [],
            "beats": [],
            "downbeats": [],
            "segments": [],
            "duration": 0.0,
            "mode": self.mode,
        }
