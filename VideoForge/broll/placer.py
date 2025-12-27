from bisect import bisect_right
import logging
from typing import Dict, List


logger = logging.getLogger(__name__)


def _map_time_to_timeline(segment_map: List, time_sec: float) -> float:
    for seg in segment_map:
        if seg.t0 <= time_sec <= seg.t1:
            return seg.timeline_start + (time_sec - seg.t0)
    return 0.0


def _build_segment_index(segment_map: List) -> tuple[List[float], List[float], List[float]]:
    starts = [float(seg.t0) for seg in segment_map]
    ends = [float(seg.t1) for seg in segment_map]
    timeline = [float(seg.timeline_start) for seg in segment_map]
    return starts, ends, timeline


def _map_time_to_timeline_fast(
    starts: List[float],
    ends: List[float],
    timeline: List[float],
    time_sec: float,
) -> float:
    idx = bisect_right(starts, float(time_sec)) - 1
    if idx < 0:
        return 0.0
    if time_sec <= ends[idx]:
        return timeline[idx] + (time_sec - starts[idx])
    return 0.0


def _time_in_segment_map(starts: List[float], ends: List[float], time_sec: float) -> bool:
    idx = bisect_right(starts, float(time_sec)) - 1
    if idx < 0:
        return False
    return float(time_sec) <= ends[idx]


def build_broll_clips(
    sentences: List[Dict],
    matches: List[Dict],
    segment_map: List,
    settings: Dict[str, Dict[str, float]],
) -> List[Dict]:
    match_by_sentence = {m["sentence_id"]: m for m in matches}
    max_duration = float(settings["broll"]["max_overlay_duration"])
    fit_mode = settings["broll"]["fit_mode"]
    crossfade = float(settings["broll"]["crossfade"])

    starts, ends, timeline_starts = _build_segment_index(segment_map)

    clips: List[Dict] = []
    for idx, sentence in enumerate(sentences, start=1):
        match = match_by_sentence.get(sentence["id"])
        if not match:
            continue
        clip_info = match.get("metadata", {}).get("clip", {})
        clip_path = clip_info.get("path")
        if not clip_path:
            continue

        t0 = float(sentence["t0"])
        t1 = float(sentence["t1"])
        if not _time_in_segment_map(starts, ends, t0) or not _time_in_segment_map(starts, ends, t1):
            logger.info("Skipping B-roll (out of range)")
            continue
        sentence_duration = t1 - t0
        timeline_start = _map_time_to_timeline_fast(
            starts, ends, timeline_starts, t0
        )
        timeline_end = _map_time_to_timeline_fast(
            starts, ends, timeline_starts, t1
        )
        timeline_duration = max(0.0, timeline_end - timeline_start)
        if timeline_duration <= 0:
            timeline_duration = sentence_duration
        duration = min(timeline_duration, max_duration)
        media_start = 0.0
        media_end = media_start + duration

        clips.append(
            {
                "id": f"broll_{idx:03d}",
                "file": clip_path,
                "media_start": media_start,
                "media_end": media_end,
                "timeline_start": timeline_start,
                "duration": duration,
                "speed": 1.0,
                "opacity": 1.0,
                "audio_enabled": False,
                "video_enabled": True,
                "match_info": {
                    "sentence_id": sentence["id"],
                    "sentence": sentence["text"],
                    "score": match.get("score", 0.0),
                    "explain": match.get("metadata", {}).get("explain", {}),
                },
                "placement": {
                    "fit_mode": fit_mode,
                    "crop": [media_start, media_end],
                    "crossfade_in": crossfade,
                    "crossfade_out": crossfade,
                },
            }
        )

    return clips
