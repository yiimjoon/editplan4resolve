from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional


logger = logging.getLogger(__name__)


class SAM3Adapter:
    def __init__(self) -> None:
        from VideoForge.adapters.sam3_subprocess import SAM3Subprocess

        self.subprocess = SAM3Subprocess()

    def generate_vision_tags(
        self,
        video_path: Path,
        prompts: Optional[List[str]] = None,
        min_frames: int = 5,
        max_frames: int = 300,
    ) -> List[str]:
        try:
            from VideoForge.config.config_manager import Config

            model_size = str(Config.get("sam3_model_size") or "large")
        except Exception:
            model_size = "large"

        if not prompts:
            prompts = [
                "person",
                "car",
                "building",
                "nature",
                "object",
            ]

        detected: List[str] = []
        for prompt in prompts:
            result = self.subprocess.segment_video(
                video_path,
                prompt,
                model_size=model_size,
                max_frames=max_frames,
            )
            if not result:
                continue
            detected_frames = int(result.get("detected_frames") or 0)
            if detected_frames >= min_frames:
                detected.append(prompt)
                logger.debug("SAM3 detected '%s' (%d frames)", prompt, detected_frames)

        logger.info("SAM3 detected tags: %s", ", ".join(detected) if detected else "none")
        return detected
