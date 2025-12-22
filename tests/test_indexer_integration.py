from pathlib import Path

import numpy as np

import base64

from VideoForge.broll.db import LibraryDB
from VideoForge.broll import indexer


def test_scan_library_uses_sampling(tmp_path: Path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    image = tmp_path / "frame.png"
    image.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
        )
    )

    captured = {}

    def _fake_ffprobe(_path: str):
        return {"width": 1920, "height": 1080, "fps": 30.0, "duration": 5.0}

    def _fake_sampling(duration: float, max_frames: int = 12, base_frames: int = 6):
        captured["duration"] = duration
        return [0.5, 1.0]

    def _fake_encode(path: str, sample_points=None, seed_hint=None):
        captured["sample_points"] = list(sample_points or [])
        return np.ones(512, dtype=np.float32)

    def _fake_image_encode(path: str, seed_hint=None):
        captured["image_path"] = path
        return np.ones(512, dtype=np.float32)

    monkeypatch.setattr(indexer, "_ffprobe_basic", _fake_ffprobe)
    monkeypatch.setattr(indexer, "smart_frame_sampling", _fake_sampling)
    monkeypatch.setattr(indexer, "encode_video_clip", _fake_encode)
    monkeypatch.setattr(indexer, "encode_image_clip", _fake_image_encode)

    db_path = tmp_path / "library.db"
    schema_path = Path("VideoForge/config/schema.sql")
    db = LibraryDB(str(db_path), schema_path=str(schema_path))

    count = indexer.scan_library(str(tmp_path), db)
    assert count == 2
    assert captured["duration"] == 5.0
    assert captured["sample_points"] == [0.5, 1.0]
    assert captured["image_path"].endswith("frame.png")
