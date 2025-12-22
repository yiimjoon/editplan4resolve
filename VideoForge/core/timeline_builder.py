import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from VideoForge.broll.placer import build_broll_clips
from VideoForge.core.segment_store import SegmentStore


def _ffprobe_video_info(path: str) -> Dict[str, float | str | List[int]]:
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
        import subprocess

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
            encoding="utf-8",
            errors="ignore",
        )
        data = json.loads(proc.stdout)
        stream = (data.get("streams") or [{}])[0]
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        fps_raw = stream.get("avg_frame_rate", "0/1")
        num, den = fps_raw.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 0.0
        duration = float(data.get("format", {}).get("duration", 0.0))
        return {
            "path": path,
            "duration": duration,
            "fps": fps,
            "resolution": [width, height],
        }
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Command timed out after 300s: {cmd[0]}") from exc
    except Exception:
        return {"path": path, "duration": 0.0, "fps": 0.0, "resolution": [0, 0]}


@dataclass
class SegmentMap:
    segment_id: int
    t0: float
    t1: float
    timeline_start: float


class TimelineBuilder:
    def __init__(self, store: SegmentStore, settings: Dict[str, Dict[str, float]]) -> None:
        self.store = store
        self.settings = settings

    def build(self, main_video_path: str) -> Dict:
        start_time = time.time()
        segments = self.store.get_segments()
        sentences = self.store.get_sentences()
        matches = self.store.get_matches()

        video_info = _ffprobe_video_info(main_video_path)
        fps = video_info.get("fps", 0.0) or self.settings.get("settings", {}).get("fps", 0.0)
        resolution = video_info.get("resolution", [0, 0])

        main_clips, segment_map = self._build_main_track(main_video_path, segments)
        broll_clips = build_broll_clips(
            sentences=sentences,
            matches=matches,
            segment_map=segment_map,
            settings=self.settings,
        )

        timeline = {
            "schema_version": str(SegmentStore.SCHEMA_VERSION),
            "pipeline_version": "0.1.0",
            "metadata": {
                "project_id": self.store.get_artifact("project_id") or "",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "main_video": {
                    "path": main_video_path,
                    "duration": video_info.get("duration", 0.0),
                    "fps": fps,
                    "timebase": None,
                    "resolution": resolution,
                },
            },
            "settings": {
                "fps": fps,
                "resolution": resolution,
                "audio_sample_rate": 48000,
            },
            "tracks": [
                {
                    "id": "V1",
                    "type": "video_main",
                    "index": 1,
                    "clips": main_clips,
                },
                {
                    "id": "V2",
                    "type": "video_broll",
                    "index": 2,
                    "clips": broll_clips,
                },
            ],
            "pipeline_info": {
                "silence_threshold_db": self.settings["silence"]["threshold_db"],
                "whisper_model": self.store.get_artifact("whisper_model") or "",
                "matching_threshold": self.settings["broll"]["matching_threshold"],
                "total_segments": len(segments),
                "total_sentences": len(sentences),
                "total_matches": len(matches),
                "processing_time_sec": round(time.time() - start_time, 3),
            },
        }
        return timeline

    def _build_main_track(
        self, main_video_path: str, segments: List[Dict]
    ) -> Tuple[List[Dict], List[SegmentMap]]:
        clips: List[Dict] = []
        segment_map: List[SegmentMap] = []
        timeline_cursor = 0.0
        for idx, seg in enumerate(segments, start=1):
            duration = max(0.0, float(seg["t1"]) - float(seg["t0"]))
            clips.append(
                {
                    "id": f"clip_{idx:03d}",
                    "file": main_video_path,
                    "media_start": float(seg["t0"]),
                    "media_end": float(seg["t1"]),
                    "timeline_start": timeline_cursor,
                    "duration": duration,
                    "speed": 1.0,
                    "audio_enabled": True,
                    "video_enabled": True,
                }
            )
            segment_map.append(
                SegmentMap(
                    segment_id=int(seg["id"]),
                    t0=float(seg["t0"]),
                    t1=float(seg["t1"]),
                    timeline_start=timeline_cursor,
                )
            )
            timeline_cursor += duration
        return clips, segment_map


def map_to_timeline(segment_map: List[SegmentMap], t0: float) -> float:
    for seg in segment_map:
        if seg.t0 <= t0 <= seg.t1:
            return seg.timeline_start + (t0 - seg.t0)
    return 0.0
