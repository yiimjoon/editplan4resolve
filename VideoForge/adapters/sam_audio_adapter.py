from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple


logger = logging.getLogger(__name__)


class SAMAudioAdapter:
    def __init__(self) -> None:
        from VideoForge.adapters.sam_audio_subprocess import SAMAudioSubprocess

        self.subprocess = SAMAudioSubprocess()

    def preprocess_audio(self, audio_path: Path) -> Optional[Path]:
        """Legacy helper for speech extraction with a default prompt."""
        return self.separate_audio(audio_path, prompt="human speech, voice, talking")

    def separate_audio(
        self,
        source_path: Path,
        prompt: str,
        output_path: Optional[Path] = None,
    ) -> Optional[Path]:
        if output_path is None:
            output_path = source_path.with_name(f"{source_path.stem}_isolated.wav")
        audio_path, temp_paths = self._prepare_audio_input(source_path)
        try:
            from VideoForge.config.config_manager import Config

            model_size = str(Config.get("sam_audio_model_size") or "large")
            timeout_sec = float(Config.get("sam_audio_timeout_sec") or 1000.0)
        except Exception:
            model_size = "large"
            timeout_sec = 1000.0

        result = self.subprocess.separate_speech(
            audio_path,
            prompt=prompt,
            model_size=model_size,
            timeout_sec=timeout_sec,
            output_path=output_path,
        )
        if not result:
            self._cleanup_temp(temp_paths)
            return None
        output_path = result.get("output_path")
        if not output_path:
            self._cleanup_temp(temp_paths)
            return None
        win_path = self._from_wsl_path(str(output_path))
        if win_path:
            self._cleanup_temp(temp_paths)
            return Path(win_path)
        self._cleanup_temp(temp_paths)
        return Path(output_path)

    def _prepare_audio_input(self, source_path: Path) -> Tuple[Path, list[str]]:
        """Ensure SAM Audio receives a WAV input; returns (audio_path, temp_paths)."""
        temp_paths: list[str] = []
        source_path = Path(source_path)
        if source_path.suffix.lower() == ".wav":
            return source_path, temp_paths

        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav", prefix="vf_audio_")
        os.close(temp_fd)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(source_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            temp_path,
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            encoding="utf-8",
            errors="ignore",
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg audio extraction failed: {proc.stderr.strip()}")
        temp_paths.append(temp_path)
        return Path(temp_path), temp_paths

    def _cleanup_temp(self, temp_paths: list[str]) -> None:
        for path in temp_paths:
            try:
                os.unlink(path)
            except Exception:
                pass

    @staticmethod
    def _from_wsl_path(path: str) -> Optional[str]:
        if path.startswith("/mnt/") and len(path) > 6:
            drive = path[5].upper()
            rest = path[6:]
            return f"{drive}:{rest.replace('/', '\\\\')}"
        return None
