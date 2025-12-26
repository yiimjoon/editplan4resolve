from bisect import bisect_left
from typing import Dict, List

import logging

from VideoForge.adapters.audio_adapter import render_ordered_segments

from VideoForge.adapters.audio_adapter import extract_silence_segments


def process_silence(
    video_path: str,
    threshold_db: float,
    min_keep: float,
    buffer_before: float,
    buffer_after: float,
) -> List[Dict[str, float]]:
    """
    Returns speech segments for the main video.
    """
    return extract_silence_segments(
        video_path=video_path,
        threshold_db=threshold_db,
        min_keep=min_keep,
        buffer_before=buffer_before,
        buffer_after=buffer_after,
    )


def render_silence_removed_with_reorder(
    video_path: str,
    sentences: List[dict],
    draft_order: List[str] | None,
    silence_segments: List[Dict[str, float]] | None,
    output_path: str,
    source_range: tuple[float, float] | None = None,
    min_duration: float = 0.5,
    tail_min_duration: float = 0.8,
    tail_ratio: float = 0.2,
    jlcut_mode: str | None = None,
    jcut_offset: float = 0.0,
    lcut_offset: float = 0.0,
) -> str:
    """Render silence-removed video following draft order."""
    logger = logging.getLogger(__name__)
    ordered_sentences = _order_sentences_by_uid(sentences, draft_order)
    ordered_segments = _build_ordered_segments(
        ordered_sentences,
        silence_segments or [],
        min_duration=min_duration,
        tail_min_duration=tail_min_duration,
        tail_ratio=tail_ratio,
    )
    logger.info(
        "Render reorder: %d sentences -> %d segments",
        len(ordered_sentences),
        len(ordered_segments),
    )
    if ordered_segments:
        total = sum(float(seg["t1"]) - float(seg["t0"]) for seg in ordered_segments)
        shortest = min(float(seg["t1"]) - float(seg["t0"]) for seg in ordered_segments)
        longest = max(float(seg["t1"]) - float(seg["t0"]) for seg in ordered_segments)
        logger.info(
            "Render reorder durations: total=%.2fs min=%.2fs max=%.2fs",
            total,
            shortest,
            longest,
        )
    return render_ordered_segments(
        video_path=video_path,
        segments=ordered_segments,
        output_path=output_path,
        source_range=source_range,
        jlcut_mode=jlcut_mode,
        jcut_offset=jcut_offset,
        lcut_offset=lcut_offset,
    )


def build_ordered_segments_for_draft(
    sentences: List[dict],
    draft_order: List[str] | None,
    min_duration: float = 0.5,
    tail_min_duration: float = 0.8,
    tail_ratio: float = 0.2,
    source_range: tuple[float, float] | None = None,
) -> List[Dict[str, float]]:
    """Build ordered segments for draft without silence filtering."""
    logger = logging.getLogger(__name__)
    ordered_sentences = _order_sentences_by_uid(sentences, draft_order)
    ordered_segments = _build_ordered_segments(
        ordered_sentences,
        [],
        min_duration=min_duration,
        tail_min_duration=tail_min_duration,
        tail_ratio=tail_ratio,
    )
    if source_range:
        clip_start, clip_end = source_range
        clipped: List[Dict[str, float]] = []
        for seg in ordered_segments:
            t0 = float(seg["t0"])
            t1 = float(seg["t1"])
            if t1 <= clip_start or t0 >= clip_end:
                continue
            t0 = max(t0, clip_start)
            t1 = min(t1, clip_end)
            if t1 <= t0:
                continue
            clipped.append({"t0": t0, "t1": t1})
        ordered_segments = clipped
    logger.info(
        "Draft segments prepared: %d sentences -> %d segments",
        len(ordered_sentences),
        len(ordered_segments),
    )
    return ordered_segments


def _order_sentences_by_uid(sentences: List[dict], draft_order: List[str] | None) -> List[dict]:
    if not draft_order:
        return sentences
    by_uid = {}
    for sentence in sentences:
        metadata = sentence.get("metadata") or {}
        uid = metadata.get("uid")
        if uid:
            by_uid[uid] = sentence
    ordered = []
    for uid in draft_order:
        if uid in by_uid:
            ordered.append(by_uid.pop(uid))
    ordered.extend(by_uid.values())
    return ordered


def _build_ordered_segments(
    sentences: List[dict],
    silence_segments: List[Dict[str, float]],
    min_duration: float = 0.5,
    tail_min_duration: float = 0.8,
    tail_ratio: float = 0.2,
) -> List[Dict[str, float]]:
    logger = logging.getLogger(__name__)
    dropped_short = 0
    dropped_tail = 0
    if not silence_segments:
        ordered = []
        for sentence in sentences:
            t0 = float(sentence.get("t0", 0.0))
            t1 = float(sentence.get("t1", 0.0))
            if t1 <= t0:
                continue
            if (t1 - t0) < min_duration:
                dropped_short += 1
                continue
            ordered.append({"t0": t0, "t1": t1})
        if dropped_short:
            logger.info("Silence render: dropped %d short sentences (< %.2fs)", dropped_short, min_duration)
        return ordered
    ordered = []
    segments_sorted = sorted(silence_segments, key=lambda seg: float(seg.get("t0", 0.0)))
    starts = [float(seg.get("t0", 0.0)) for seg in segments_sorted]
    ends = [float(seg.get("t1", 0.0)) for seg in segments_sorted]
    seg_count = len(segments_sorted)
    for sentence in sentences:
        s0 = float(sentence.get("t0", 0.0))
        s1 = float(sentence.get("t1", 0.0))
        if s1 <= s0:
            continue
        sentence_dur = s1 - s0
        overlaps: List[Dict[str, float]] = []
        idx = bisect_left(ends, s0)
        while idx < seg_count and starts[idx] < s1:
            t0 = max(s0, starts[idx])
            t1 = min(s1, ends[idx])
            if t1 <= t0:
                idx += 1
                continue
            if (t1 - t0) < min_duration:
                dropped_short += 1
                idx += 1
                continue
            overlaps.append({"t0": t0, "t1": t1})
            idx += 1
        if overlaps:
            last = overlaps[-1]
            last_dur = last["t1"] - last["t0"]
            tail_ratio_value = last_dur / sentence_dur if sentence_dur > 0 else 1.0
            if last_dur < tail_min_duration or tail_ratio_value < tail_ratio:
                dropped_tail += 1
                text = (sentence.get("text") or "").strip()
                snippet = text[-24:] if text else ""
                logger.info(
                    "Silence render: dropped tail (%.2fs, %.0f%%) uid=%s text='...%s'",
                    last_dur,
                    tail_ratio_value * 100.0,
                    (sentence.get("metadata") or {}).get("uid"),
                    snippet,
                )
                overlaps = overlaps[:-1]
        ordered.extend(overlaps)
    if dropped_short:
        logger.info("Silence render: dropped %d short segments (< %.2fs)", dropped_short, min_duration)
    if dropped_tail:
        logger.info("Silence render: dropped %d tail fragments (< %.2fs or < %.0f%%)", dropped_tail, tail_min_duration, tail_ratio * 100.0)
    return ordered
