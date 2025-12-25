"""B-roll source and media type configuration."""
from __future__ import annotations

from typing import List


class BrollSourceConfig:
    """B-roll generation source configuration."""

    SOURCE_OPTIONS = [
        "library_only",
        "library_stock",
        "library_ai",
        "library_stock_ai",
        "stock_only",
        "ai_only",
    ]

    MEDIA_TYPE_OPTIONS = [
        "auto",
        "prefer_video",
        "image_only",
    ]

    MEDIA_TYPE_DEFAULTS = {
        "pexels": "video",
        "unsplash": "image",
        "comfyui": "image",
        "library": "both",
    }

    def __init__(
        self,
        source_mode: str = "stock_only",
        prefer_media_type: str = "auto",
        max_scenes: int = 15,
    ) -> None:
        self.source_mode = source_mode
        self.prefer_media_type = prefer_media_type
        self.max_scenes = max_scenes

    def get_allowed_media_types(self, source: str) -> List[str]:
        """Return allowed media types for a given source."""
        available = {
            "pexels": ["video", "image"],
            "unsplash": ["image"],
            "comfyui": ["image"],
            "library": ["video", "image"],
        }
        source_types = available.get(source, [])

        if self.prefer_media_type == "auto":
            return source_types
        if self.prefer_media_type == "prefer_video":
            return ["video"] if "video" in source_types else []
        return ["image"] if "image" in source_types else []

    @classmethod
    def from_config(cls) -> "BrollSourceConfig":
        """Load configuration from Config."""
        from VideoForge.config.config_manager import Config

        source_mode = str(Config.get("broll_source_mode") or "stock_only").strip()
        prefer_media_type = str(Config.get("broll_prefer_media_type") or "auto").strip()
        raw_max = Config.get("broll_max_scenes")
        try:
            max_scenes = int(raw_max) if raw_max is not None else 15
        except (TypeError, ValueError):
            max_scenes = 15

        if source_mode not in cls.SOURCE_OPTIONS:
            source_mode = "stock_only"
        if prefer_media_type not in cls.MEDIA_TYPE_OPTIONS:
            prefer_media_type = "auto"

        return cls(
            source_mode=source_mode,
            prefer_media_type=prefer_media_type,
            max_scenes=max_scenes,
        )
