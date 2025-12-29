from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from VideoForge.adapters.audio_sync import _extract_pcm

logger = logging.getLogger(__name__)


class VoiceDetector:
    """Detect speech ranges using Silero VAD."""

    _model = None
    _utils = None
    _load_error: str | None = None

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 100,
        speech_pad_ms: int = 30,
    ) -> None:
        self.sample_rate = max(1000, int(sample_rate))
        self.threshold = float(threshold)
        self.min_speech_ms = max(0, int(min_speech_ms))
        self.min_silence_ms = max(0, int(min_silence_ms))
        self.speech_pad_ms = max(0, int(speech_pad_ms))

    @classmethod
    def is_available(cls) -> bool:
        if cls._load_error:
            return False
        try:
            return importlib.util.find_spec("torch") is not None
        except Exception:
            return False

    @classmethod
    def _load_model(cls) -> bool:
        if cls._model is not None and cls._utils is not None:
            return True
        if cls._load_error:
            return False
        try:
            import torch

            torch.set_num_threads(1)
            cls._model, cls._utils = torch.hub.load(
                "snakers4/silero-vad",
                "silero_vad",
                trust_repo=True,
            )
            return True
        except Exception as exc:
            cls._load_error = str(exc)
            logger.warning("Silero VAD unavailable: %s", exc)
            return False

    def detect_voice(self, video_path: Path) -> List[Tuple[float, float]]:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(path)
        if not self._load_model():
            return []
        try:
            pcm = _extract_pcm(path, self.sample_rate)
        except Exception as exc:
            logger.warning("Audio extraction failed: %s", exc)
            return []
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return []
        samples /= 32768.0
        get_speech_timestamps = self._utils[0]
        try:
            timestamps = get_speech_timestamps(
                samples,
                self._model,
                sampling_rate=self.sample_rate,
                threshold=self.threshold,
                min_speech_duration_ms=self.min_speech_ms,
                min_silence_duration_ms=self.min_silence_ms,
                speech_pad_ms=self.speech_pad_ms,
            )
        except Exception as exc:
            logger.warning("Voice detection failed: %s", exc)
            return []
        return [
            (float(ts["start"]) / self.sample_rate, float(ts["end"]) / self.sample_rate)
            for ts in timestamps
        ]
