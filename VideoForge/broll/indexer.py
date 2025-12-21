import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List

from VideoForge.adapters.embedding_adapter import encode_video_clip
from VideoForge.broll.db import LibraryDB


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}


def smart_frame_sampling(duration: float, max_frames: int = 12, base_frames: int = 6) -> List[float]:
    if duration <= 0:
        return [0.0]
    if duration < 3.0:
        return _unique_sorted([i * duration / 2 for i in range(3)], duration)

    excl = min(2.0, duration * 0.15)

    if 3.0 <= duration <= 10.0:
        start_anchor = min(excl, duration * 0.1)
        end_anchor = max(duration - excl, duration * 0.9)
        anchors = [
            start_anchor,
            duration * 0.3,
            duration * 0.5,
            duration * 0.7,
            end_anchor,
        ]
        return _unique_sorted(anchors, duration)

    target_frames = min(max_frames, max(base_frames, int(duration / 3)))
    inner_count = max(0, target_frames - 2)
    inner_start = excl
    inner_end = max(excl, duration - excl)

    inner_points: List[float] = []
    if inner_count > 0:
        step = (inner_end - inner_start) / max(inner_count, 1)
        for i in range(inner_count):
            inner_points.append(inner_start + i * step)

    anchors = [min(excl, duration * 0.1), max(duration - excl, duration * 0.9)]
    return _unique_sorted(anchors + inner_points, duration)


def _unique_sorted(points: Iterable[float], duration: float) -> List[float]:
    clamped = [min(max(0.0, p), duration) for p in points]
    unique = sorted(set(round(p, 3) for p in clamped))
    return unique


def _ffprobe_basic(path: str) -> Dict[str, float | int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(proc.stdout)
        stream = (data.get("streams") or [{}])[0]
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        fps_raw = stream.get("avg_frame_rate", "0/1")
        num, den = fps_raw.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 0.0
        duration = float(data.get("format", {}).get("duration", 0.0))
        return {"width": width, "height": height, "fps": fps, "duration": duration}
    except Exception:
        return {"width": 0, "height": 0, "fps": 0.0, "duration": 0.0}


def scan_library(library_dir: str, db: LibraryDB) -> int:
    library_path = Path(library_dir)
    if not library_path.exists():
        raise FileNotFoundError(library_dir)

    count = 0
    for path in library_path.rglob("*"):
        if path.suffix.lower() not in VIDEO_EXTS:
            continue
        info = _ffprobe_basic(str(path))
        folder_tags = " ".join(part for part in path.parent.parts if part)
        metadata = {
            "path": str(path),
            "duration": info["duration"],
            "fps": info["fps"],
            "width": info["width"],
            "height": info["height"],
            "folder_tags": folder_tags,
            "vision_tags": "",
            "description": "",
        }
        embedding = encode_video_clip(str(path))
        db.upsert_clip(metadata, embedding)
        count += 1
    return count
