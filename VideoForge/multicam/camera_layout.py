"""Camera layout helpers for gaze-aware multicam scoring."""

from __future__ import annotations

from typing import List

from VideoForge.config.config_manager import Config


def _normalize_layout(value) -> List[str]:
    if isinstance(value, str):
        raw = value.replace(" ", "").replace(";", ",")
        if "," in raw:
            parts = [p.strip().upper() for p in raw.split(",") if p.strip()]
        else:
            parts = [c.upper() for c in raw if c.strip()]
        return [p for p in parts if p in ("L", "C", "R")]
    if isinstance(value, list):
        parts = []
        for item in value:
            if not item:
                continue
            if isinstance(item, str):
                token = item.strip().upper()
                if token in ("L", "C", "R"):
                    parts.append(token)
            elif isinstance(item, (int, float)):
                token = "C" if int(item) == 0 else ("L" if int(item) < 0 else "R")
                parts.append(token)
        return parts
    return []


def resolve_layout(track_count: int) -> List[str]:
    if track_count <= 0:
        return []
    configured = _normalize_layout(Config.get("multicam_camera_layout", []))
    if len(configured) == track_count:
        return configured

    if track_count == 1:
        return ["C"]
    if track_count == 2:
        return ["L", "R"]
    if track_count == 3:
        return ["L", "C", "R"]

    layout = ["C"] * track_count
    layout[0] = "L"
    layout[-1] = "R"
    return layout


def _yaw_map(track_count: int) -> dict[str, float]:
    if track_count == 2:
        return {"L": -30.0, "C": 0.0, "R": 30.0}
    if track_count == 3:
        return {"L": -35.0, "C": 0.0, "R": 35.0}
    return {"L": -35.0, "C": 0.0, "R": 35.0}


def get_track_yaw(track_index: int, track_count: int) -> float:
    layout = resolve_layout(track_count)
    if not layout or track_index < 1 or track_index > len(layout):
        return 0.0
    slot = layout[track_index - 1]
    yaw_map = _yaw_map(track_count)
    return float(yaw_map.get(slot, 0.0))
