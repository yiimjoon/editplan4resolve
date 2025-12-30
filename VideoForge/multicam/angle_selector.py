"""Select the best camera angle per segment based on score rules."""

from __future__ import annotations

from typing import Dict, List, Optional

from VideoForge.config.config_manager import Config


class AngleSelector:
    """Score-based angle selection engine."""

    def __init__(self) -> None:
        self.min_hold_sec = float(Config.get("multicam_min_hold_sec", 2.0))
        self.max_repeat = int(Config.get("multicam_max_repeat", 3))
        self.closeup_weight = float(Config.get("multicam_closeup_weight", 0.3))
        self.wide_weight = float(Config.get("multicam_wide_weight", 0.2))

    def select_angles(
        self,
        segments: List[Dict],
        segment_tags: Optional[List[str]] = None,
    ) -> List[Dict]:
        selections: List[Dict] = []
        prev_angle = None
        repeat_count = 0

        for idx, segment in enumerate(segments):
            tag = None
            if segment_tags and idx < len(segment_tags):
                tag = str(segment_tags[idx]).strip().lower()

            raw_scores = segment.get("scores") or {}
            if not isinstance(raw_scores, dict) or not raw_scores:
                continue

            angle_scores: Dict[int, float] = {}
            angle_metric_map: Dict[int, Dict[str, float]] = {}
            for angle_idx, metrics in raw_scores.items():
                try:
                    angle_key = int(angle_idx)
                except Exception:
                    continue
                if not isinstance(metrics, dict):
                    metrics = {}
                angle_metric_map[angle_key] = metrics
                angle_scores[angle_key] = self._calculate_composite_score(metrics, tag)

            if not angle_scores:
                continue

            best_angle = max(angle_scores, key=angle_scores.get)

            if best_angle == prev_angle:
                repeat_count += 1
                if repeat_count >= self.max_repeat:
                    sorted_angles = sorted(
                        angle_scores.items(), key=lambda item: item[1], reverse=True
                    )
                    if len(sorted_angles) > 1:
                        best_angle = sorted_angles[1][0]
                        repeat_count = 1
            else:
                repeat_count = 1

            duration = float(segment.get("end_sec", 0.0)) - float(segment.get("start_sec", 0.0))
            if duration < self.min_hold_sec and selections:
                selections[-1]["end_sec"] = float(segment.get("end_sec", selections[-1]["end_sec"]))
                continue

            metrics = angle_metric_map.get(best_angle, {})
            selections.append(
                {
                    "start_sec": float(segment.get("start_sec", 0.0)),
                    "end_sec": float(segment.get("end_sec", 0.0)),
                    "angle_index": best_angle,
                    "angle_score": float(angle_scores[best_angle]),
                    "reason": self._generate_reason(metrics, tag),
                }
            )
            prev_angle = best_angle

        return self._postprocess_short_segments(selections)

    def _calculate_composite_score(
        self,
        scores: Dict[str, float],
        tag: Optional[str] = None,
    ) -> float:
        weights = {
            "sharpness": 0.4,
            "stability": 0.3,
            "motion": 0.2,
            "face_score": 0.1,
        }

        if tag == "speaking":
            weights["face_score"] += self.closeup_weight
        elif tag == "action":
            weights["motion"] += self.wide_weight

        total = sum(weights.values())
        if total <= 0:
            total = 1.0
        weights = {key: value / total for key, value in weights.items()}

        composite = 0.0
        for key, weight in weights.items():
            try:
                value = float(scores.get(key, 0.0) or 0.0)
            except Exception:
                value = 0.0
            composite += value * weight
        return float(composite)

    @staticmethod
    def _generate_reason(scores: Dict[str, float], tag: Optional[str] = None) -> str:
        if not scores:
            return "No score data available"
        top_metric = max(scores, key=scores.get)
        parts = [f"Best {top_metric} ({scores.get(top_metric, 0.0):.2f})"]
        if tag:
            parts.append(f"[{tag}]")
        if float(scores.get("face_score", 0.0) or 0.0) > 0.7:
            parts.append("+ centered face")
        return " ".join(parts)

    def _postprocess_short_segments(self, selections: List[Dict]) -> List[Dict]:
        if not selections:
            return []

        final: List[Dict] = []
        pending: Optional[Dict] = None

        for sel in selections:
            start_sec = float(sel.get("start_sec", 0.0))
            end_sec = float(sel.get("end_sec", 0.0))
            duration = end_sec - start_sec

            if duration < self.min_hold_sec:
                if final:
                    final[-1]["end_sec"] = end_sec
                else:
                    if pending is None:
                        pending = dict(sel)
                    else:
                        pending["end_sec"] = end_sec
                continue

            if pending is not None:
                sel = dict(sel)
                sel["start_sec"] = float(pending.get("start_sec", start_sec))
                pending = None

            final.append(sel)

        if pending is not None:
            if final:
                final[-1]["end_sec"] = float(pending.get("end_sec", final[-1]["end_sec"]))
            else:
                final.append(pending)

        return final
