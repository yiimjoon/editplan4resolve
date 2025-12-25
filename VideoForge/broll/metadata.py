from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

SCHEMA_ID = "videoforge.broll.v1"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _split_keywords(text: str) -> List[str]:
    if not text:
        return []
    parts = [p.strip() for p in text.replace(",", " ").split()]
    return [p for p in parts if p]


def _normalize_keywords(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return _split_keywords(value)
    if isinstance(value, Iterable):
        items: List[str] = []
        for item in value:
            item_text = _coerce_text(item)
            if item_text:
                items.extend(_split_keywords(item_text))
        return items
    return []


def _merge_unique(items: Iterable[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for item in items:
        token = _coerce_text(item)
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        merged.append(token)
    return merged


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class BrollMetadata:
    source: str
    keywords: List[str] = field(default_factory=list)
    description: str = ""
    visual_elements: List[str] = field(default_factory=list)
    project_name: str = ""
    generated_from_text: str = ""
    created_at: str = ""
    duration: Optional[float] = None
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def apply_context(self, project_name: str, generated_from_text: str) -> None:
        if project_name and not self.project_name:
            self.project_name = project_name
        if generated_from_text and not self.generated_from_text:
            self.generated_from_text = generated_from_text
        if not self.created_at:
            self.created_at = now_iso()

    def merge_keywords(self, extra: Iterable[str]) -> None:
        self.keywords = _merge_unique(list(self.keywords) + list(extra))
        if not self.visual_elements:
            self.visual_elements = list(self.keywords)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "schema": SCHEMA_ID,
            "source": _coerce_text(self.source),
            "keywords": _merge_unique(self.keywords),
            "description": _coerce_text(self.description),
            "visual_elements": _merge_unique(self.visual_elements or []),
            "project_name": _coerce_text(self.project_name),
            "generated_from_text": _coerce_text(self.generated_from_text),
            "created_at": _coerce_text(self.created_at),
            "raw_metadata": self.raw_metadata or {},
        }
        if self.duration is not None:
            payload["duration"] = float(self.duration)
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BrollMetadata":
        return cls(
            source=_coerce_text(data.get("source")),
            keywords=_normalize_keywords(data.get("keywords")),
            description=_coerce_text(data.get("description")),
            visual_elements=_normalize_keywords(data.get("visual_elements")),
            project_name=_coerce_text(data.get("project_name")),
            generated_from_text=_coerce_text(data.get("generated_from_text")),
            created_at=_coerce_text(data.get("created_at")),
            duration=_coerce_float(data.get("duration")),
            raw_metadata=data.get("raw_metadata") or {},
        )


def coerce_metadata_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not raw:
        return {}
    if raw.get("schema") == SCHEMA_ID:
        return raw
    meta = BrollMetadata(
        source=_coerce_text(raw.get("source")),
        keywords=_normalize_keywords(raw.get("keywords") or raw.get("tags")),
        description=_coerce_text(raw.get("description") or raw.get("prompt")),
        visual_elements=_normalize_keywords(raw.get("visual_elements")),
        project_name=_coerce_text(raw.get("project_name") or raw.get("project")),
        generated_from_text=_coerce_text(raw.get("generated_from_text") or raw.get("sentence")),
        created_at=_coerce_text(raw.get("created_at")),
        duration=_coerce_float(raw.get("duration")),
        raw_metadata=raw,
    )
    if not meta.created_at:
        meta.created_at = now_iso()
    if not meta.visual_elements:
        meta.visual_elements = list(meta.keywords)
    return meta.to_dict()
