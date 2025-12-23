"""Domain errors for VideoForge."""

from __future__ import annotations

from typing import Any, Dict


class VideoForgeError(Exception):
    """Base exception for VideoForge domain errors."""

    def __init__(self, message: str, *, details: Dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class AnalysisError(VideoForgeError):
    """Analysis pipeline failed."""


class MatchError(VideoForgeError):
    """B-roll matching failed."""


class TimelineError(VideoForgeError):
    """Timeline operations failed."""


class ConfigError(VideoForgeError):
    """Configuration errors."""
