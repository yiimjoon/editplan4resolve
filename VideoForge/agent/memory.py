"""Optional long-term memory store placeholder (Phase 10 Week 2+)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class MemoryRecord:
    key: str
    value: Any


class MemoryStore:
    """Minimal in-memory store used for scaffolding."""

    def __init__(self) -> None:
        self._data: Dict[str, MemoryRecord] = {}

    def put(self, key: str, value: Any) -> None:
        self._data[key] = MemoryRecord(key=key, value=value)

    def get(self, key: str) -> Optional[Any]:
        record = self._data.get(key)
        return record.value if record else None

