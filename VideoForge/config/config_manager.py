"""Persistent user configuration manager for VideoForge plugin."""

from __future__ import annotations

import json
import logging
import os
import shutil
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
        legacy_path = Path(__file__).resolve().parents[1] / "data" / "user_config.json"
        self._config_path = self._resolve_config_path()
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._config_path.exists() and legacy_path.exists():
            try:
                shutil.copy2(legacy_path, self._config_path)
                logger.info("Migrated legacy config to %s", self._config_path)
            except Exception as exc:
                logger.warning("Failed to migrate legacy config: %s", exc)
        self._load()
        self._migrate_library_config()
        self._ensure_llm_defaults()
        self._ensure_audio_sync_defaults()
        self._ensure_multicam_defaults()
        self._ensure_agent_defaults()

    @staticmethod
    def _resolve_config_path() -> Path:
        appdata = os.environ.get("APPDATA")
        if appdata:
            return (
                Path(appdata)
                / "Blackmagic Design"
                / "DaVinci Resolve"
                / "Support"
                / "VideoForge"
                / "user_config.json"
            )
        return Path(__file__).resolve().parents[1] / "data" / "user_config.json"

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

    def _migrate_library_config(self) -> None:
        changed = False
        legacy_path = self._config.pop("library_db_path", None)
        if legacy_path and not self._config.get("global_library_db_path"):
            self._config["global_library_db_path"] = legacy_path
            changed = True
        if "local_library_db_path" not in self._config:
            default_local = self._default_local_library_path()
            if default_local:
                self._config["local_library_db_path"] = default_local
                changed = True
        if changed:
            self._save()

    def _ensure_audio_sync_defaults(self) -> None:
        defaults = {
            "audio_sync_threshold_db": -40.0,
            "audio_sync_min_silence": 0.3,
            "audio_sync_max_offset": 30.0,
            "audio_sync_resolution": 0.1,
            "audio_sync_min_confidence": 0.5,
            "audio_sync_normalize": True,
            "audio_sync_subprocess_mode": "auto",
            "audio_sync_python_exe": "",
            "audio_sync_voice_threshold": 0.5,
            "audio_sync_voice_min_speech_ms": 250,
            "audio_sync_voice_min_silence_ms": 100,
            "audio_sync_voice_pad_ms": 30,
            "audio_sync_content_feature": "chroma",
            "audio_sync_content_sample_rate": 22050,
            "audio_sync_content_hop_length": 512,
            "audio_sync_content_n_fft": 2048,
        }
        changed = False
        for key, value in defaults.items():
            if key not in self._config:
                self._config[key] = value
                changed = True
        if changed:
            self._save()

    def _ensure_llm_defaults(self) -> None:
        defaults = {
            "llm_max_tokens": 4096,
        }
        changed = False
        for key, value in defaults.items():
            if key not in self._config:
                self._config[key] = value
                changed = True
        if changed:
            self._save()

    def _ensure_multicam_defaults(self) -> None:
        defaults = {
            "multicam_max_segment_sec": 10.0,
            "multicam_min_hold_sec": 2.0,
            "multicam_max_repeat": 3,
            "multicam_closeup_weight": 0.3,
            "multicam_wide_weight": 0.2,
            "multicam_face_detector": "opencv_dnn",
            "multicam_face_model_dir": "",
            "multicam_edl_output_dir": "",
            "multicam_boundary_mode": "hybrid",
        }
        changed = False
        for key, value in defaults.items():
            if key not in self._config:
                self._config[key] = value
                changed = True
        if changed:
            self._save()

    def _ensure_agent_defaults(self) -> None:
        defaults = {
            "agent_api_key": "",
            "agent_mode": "approve_required",
            "agent_model": "gemini-2.0-flash-exp",
            "agent_max_tokens": 4096,
            "agent_max_auto_actions": 5,
        }
        changed = False
        for key, value in defaults.items():
            if key not in self._config:
                self._config[key] = value
                changed = True
        if changed:
            self._save()

    def _default_local_library_path(self) -> str:
        project_name = os.environ.get("VIDEOFORGE_PROJECT_NAME", "").strip() or None
        if not project_name:
            try:
                from VideoForge.integrations.resolve_api import ResolveAPI

                resolve_api = ResolveAPI()
                project = resolve_api.get_current_project()
                project_name = project.GetName() if project else None
            except Exception:
                project_name = None
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (project_name or "default"))
        base = Path(__file__).resolve().parents[1] / "data" / "projects"
        base.mkdir(parents=True, exist_ok=True)
        return str(base / f"{safe}_local.db")

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
        if key == "library_db_path":
            key = "global_library_db_path"
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a config value and persist to disk."""
        if key == "library_db_path":
            key = "global_library_db_path"
        self._config[key] = value
        self._save()

    def has(self, key: str) -> bool:
        """Check if a config key exists."""
        return key in self._config


Config = ConfigManager()
