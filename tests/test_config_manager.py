"""Tests for ConfigManager."""

from pathlib import Path

from VideoForge.config.config_manager import ConfigManager


def test_config_persistence(tmp_path: Path, monkeypatch):
    """Test that config persists across instances."""
    config_file = tmp_path / "user_config.json"

    def mock_init(self):
        self._config_path = config_file
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    monkeypatch.setattr(ConfigManager, "_initialize", mock_init)

    config1 = ConfigManager()
    config1.set("broll_dir", "/path/to/broll")
    assert config1.get("broll_dir") == "/path/to/broll"
    assert config_file.exists()

    ConfigManager._instance = None
    config2 = ConfigManager()
    assert config2.get("broll_dir") == "/path/to/broll"


def test_config_defaults():
    """Test default value handling."""
    config = ConfigManager()
    assert config.get("nonexistent_key", "default") == "default"
    assert not config.has("nonexistent_key")
