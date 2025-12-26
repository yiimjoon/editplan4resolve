from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


class SAMAudioAdapter:
    def __init__(self) -> None:
        from VideoForge.adapters.sam_audio_subprocess import SAMAudioSubprocess

        self.subprocess = SAMAudioSubprocess()

    def preprocess_audio(self, audio_path: Path) -> Optional[Path]:
        try:
            from VideoForge.config.config_manager import Config

            model_size = str(Config.get("sam_audio_model_size") or "large")
        except Exception:
            model_size = "large"

        result = self.subprocess.separate_speech(
            audio_path,
            model_size=model_size,
        )
        if not result:
            return None
        output_path = result.get("output_path")
        if not output_path:
            return None
        win_path = self._from_wsl_path(str(output_path))
        if win_path:
            return Path(win_path)
        return Path(output_path)

    @staticmethod
    def _from_wsl_path(path: str) -> Optional[str]:
        if path.startswith("/mnt/") and len(path) > 6:
            drive = path[5].upper()
            rest = path[6:]
            return f"{drive}:{rest.replace('/', '\\\\')}"
        return None
