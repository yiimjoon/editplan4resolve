"""Beat-aligned clip placement utilities."""

from __future__ import annotations

from typing import Dict, List

from VideoForge.config.config_manager import Config


class BeatPlacer:
    """Arrange clips based on beat timestamps."""

    def __init__(self) -> None:
        self.mode = str(Config.get("beat_placement_mode", "on_beat")).strip().lower()
        self.min_clip = float(Config.get("beat_placement_min_clip_sec", 0.25))
        self.max_clip = float(Config.get("beat_placement_max_clip_sec", 2.0))
        self.transition_type = str(
            Config.get("beat_placement_transition_type", "cut")
        ).strip().lower()
        self.crossfade_sec = float(Config.get("beat_placement_crossfade_sec", 0.1))
        self.jl_cut_sec = float(Config.get("beat_placement_j_l_cut_sec", 0.15))

    def place_clips_on_beat(
        self,
        clips: List[Dict],
        beat_data: Dict,
        placement_mode: str | None = None,
    ) -> List[Dict]:
        if not clips:
            return []
        mode = str(placement_mode or self.mode or "on_beat").strip().lower()
        beats = [float(x) for x in (beat_data.get("beats") or [])]
        downbeats = [float(x) for x in (beat_data.get("downbeats") or [])]
        segments = beat_data.get("segments") or []

        if mode == "on_downbeat" and downbeats:
            times = downbeats
        elif mode == "on_segment" and segments:
            return self._place_on_segments(clips, segments)
        else:
            times = beats or downbeats

        if not times:
            return []

        return self._place_on_times(clips, times, mode == "fill_gaps")

    def _place_on_segments(self, clips: List[Dict], segments: List[Dict]) -> List[Dict]:
        placements: List[Dict] = []
        for idx, segment in enumerate(segments):
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start))
            duration = max(self.min_clip, min(self.max_clip, end - start))
            if duration <= 0:
                continue
            clip = clips[idx % len(clips)]
            placements.append(
                self._placement_for_clip(clip, start, start + duration)
            )
        return placements

    def _place_on_times(
        self,
        clips: List[Dict],
        times: List[float],
        fill_gaps: bool,
    ) -> List[Dict]:
        placements: List[Dict] = []
        for idx in range(len(times)):
            start = float(times[idx])
            end = float(times[idx + 1]) if idx + 1 < len(times) else start + self.max_clip
            if end <= start:
                end = start + self.min_clip
            duration = max(self.min_clip, min(self.max_clip, end - start))
            if fill_gaps and duration > self.max_clip:
                duration = self.max_clip
            clip = clips[idx % len(clips)]
            placements.append(self._placement_for_clip(clip, start, start + duration))

            if fill_gaps and end - start > self.max_clip:
                cursor = start + duration
                while cursor + self.min_clip < end:
                    next_end = min(cursor + self.max_clip, end)
                    clip = clips[(idx + len(placements)) % len(clips)]
                    placements.append(
                        self._placement_for_clip(clip, cursor, next_end)
                    )
                    cursor = next_end

        return placements

    def _placement_for_clip(self, clip: Dict, start: float, end: float) -> Dict:
        transition = self.transition_type
        transition_sec = 0.0
        if transition in ("fade", "crossfade"):
            transition_sec = self.crossfade_sec
        elif transition in ("j-cut", "l-cut", "jcut", "lcut"):
            transition_sec = self.jl_cut_sec
            transition = "j-cut" if "j" in transition else "l-cut"
        else:
            transition = "cut"
        return {
            "clip": clip,
            "start_sec": float(start),
            "end_sec": float(end),
            "transition": transition,
            "transition_sec": float(transition_sec),
        }

