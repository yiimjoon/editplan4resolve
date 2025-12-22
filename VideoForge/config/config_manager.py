"""Persistent user configuration manager for VideoForge plugin."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages persistent user settings stored in JSON file."""

    _instance = None
    _config_path: Path | None = None
    _config: dict = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize config path and load existing settings."""
        self._config_path = Path(__file__).resolve().parents[1] / "data" / "user_config.json"
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        """Load config from JSON file."""
        if self._config_path and self._config_path.exists():
            try:
                self._config = json.loads(self._config_path.read_text(encoding="utf-8"))
                logger.info("Loaded user config from %s", self._config_path)
            except Exception as exc:
                logger.warning("Failed to load config, using defaults: %s", exc)
                self._config = {}
        else:
            self._config = {}

    def _save(self) -> None:
        """Save config to JSON file."""
        if not self._config_path:
            return
        try:
            self._config_path.write_text(
                json.dumps(self._config, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.debug("Saved user config to %s", self._config_path)
        except Exception as exc:
            logger.error("Failed to save config: %s", exc)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a config value and persist to disk."""
        self._config[key] = value
        self._save()

    def has(self, key: str) -> bool:
        """Check if a config key exists."""
        return key in self._config


Config = ConfigManager()
