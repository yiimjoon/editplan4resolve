from typing import Dict, List


def _map_sentence_to_timeline(segment_map: List, t0: float) -> float:
    for seg in segment_map:
        if seg.t0 <= t0 <= seg.t1:
            return seg.timeline_start + (t0 - seg.t0)
    return 0.0


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

    clips: List[Dict] = []
    for idx, sentence in enumerate(sentences, start=1):
        match = match_by_sentence.get(sentence["id"])
        if not match:
            continue
        clip_info = match.get("metadata", {}).get("clip", {})
        clip_path = clip_info.get("path")
        if not clip_path:
            continue

        sentence_duration = float(sentence["t1"]) - float(sentence["t0"])
        duration = min(sentence_duration, max_duration)
        media_start = 0.0
        media_end = media_start + duration
        timeline_start = _map_sentence_to_timeline(segment_map, float(sentence["t0"]))

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
