"""Beat-aware B-roll matcher helpers."""

from __future__ import annotations

from typing import Dict, List

from VideoForge.broll.matcher import BrollMatcher


class BeatBrollMatcher(BrollMatcher):
    """Extend BrollMatcher with beat-aligned metadata."""

    def match_with_beats(
        self,
        sentences: List[Dict],
        beat_data: Dict,
    ) -> List[Dict]:
        beat_times = [float(x) for x in (beat_data.get("beats") or [])]
        downbeats = [float(x) for x in (beat_data.get("downbeats") or [])]
        enriched: List[Dict] = []
        for sentence in sentences:
            t0 = float(sentence.get("t0", 0.0))
            t1 = float(sentence.get("t1", t0))
            beats_in_range = [b for b in beat_times if t0 <= b <= t1]
            downbeats_in_range = [b for b in downbeats if t0 <= b <= t1]
            payload = dict(sentence)
            payload["beat_points"] = beats_in_range
            payload["downbeats"] = downbeats_in_range
            enriched.append(payload)
        matches = super().match(enriched)
        for match in matches:
            sid = match.get("sentence_id") or match.get("sentenceId")
            if sid is None:
                continue
            for sentence in enriched:
                if str(sentence.get("id")) == str(sid):
                    match["beat_points"] = sentence.get("beat_points", [])
                    match["downbeats"] = sentence.get("downbeats", [])
                    break
        return matches

