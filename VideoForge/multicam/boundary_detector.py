"""Sentence-boundary based cut detector."""

from __future__ import annotations

import logging
import re
from typing import List

from VideoForge.config.config_manager import Config

logger = logging.getLogger(__name__)

_SENTENCE_END_RE = re.compile(r"[.!?…。！？]+$")


class BoundaryDetector:
    """Generate cut boundaries from transcript segments."""

    def __init__(self, max_segment_sec: float | None = None) -> None:
        default_max = float(Config.get("multicam_max_segment_sec", 10.0))
        self.max_segment_sec = (
            float(max_segment_sec) if max_segment_sec is not None else default_max
        )

    def extract_from_transcript(
        self,
        segments: List[dict],
        mode: str = "sentence",
    ) -> List[float]:
        """
        Return sorted cut timestamps (seconds), including 0.0 and last end.

        Modes:
        - sentence: cuts at sentence-ending punctuation.
        - fixed: cuts at max_segment_sec intervals.
        - hybrid: sentence cuts, but split if gap > max_segment_sec.
        """
        if not segments:
            return [0.0]

        mode = str(mode or "sentence").strip().lower()
        if mode not in {"sentence", "fixed", "hybrid"}:
            mode = "sentence"

        total_end = max(float(seg.get("t1", 0.0)) for seg in segments)
        if total_end <= 0:
            return [0.0]

        if mode == "fixed":
            return self._fixed_boundaries(total_end)

        boundaries = [0.0]
        for seg in segments:
            text = str(seg.get("text", "")).strip()
            end = float(seg.get("t1", 0.0))
            if end <= boundaries[-1]:
                continue
            if not text or _SENTENCE_END_RE.search(text):
                boundaries.append(end)

        if boundaries[-1] < total_end:
            boundaries.append(total_end)

        if mode == "hybrid":
            boundaries = self._insert_max_gaps(boundaries, total_end)

        return sorted({round(ts, 3) for ts in boundaries})

    def _fixed_boundaries(self, total_end: float) -> List[float]:
        if self.max_segment_sec <= 0:
            return [0.0, total_end]
        cuts = [0.0]
        t = self.max_segment_sec
        while t < total_end:
            cuts.append(t)
            t += self.max_segment_sec
        cuts.append(total_end)
        return cuts

    def _insert_max_gaps(self, boundaries: List[float], total_end: float) -> List[float]:
        if self.max_segment_sec <= 0:
            return boundaries
        expanded = [boundaries[0]]
        for idx in range(1, len(boundaries)):
            prev = expanded[-1]
            current = boundaries[idx]
            gap = current - prev
            if gap > self.max_segment_sec:
                t = prev + self.max_segment_sec
                while t < current:
                    expanded.append(t)
                    t += self.max_segment_sec
            expanded.append(current)
        if expanded[-1] < total_end:
            expanded.append(total_end)
        return expanded
