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
) -> str:
    """Render silence-removed video following draft order."""
    logger = logging.getLogger(__name__)
    ordered_sentences = _order_sentences_by_uid(sentences, draft_order)
    ordered_segments = _build_ordered_segments(
        ordered_sentences,
        silence_segments or [],
        min_duration=min_duration,
    )
    logger.info(
        "Render reorder: %d sentences -> %d segments",
        len(ordered_sentences),
        len(ordered_segments),
    )
    return render_ordered_segments(
        video_path=video_path,
        segments=ordered_segments,
        output_path=output_path,
        source_range=source_range,
    )


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
) -> List[Dict[str, float]]:
    if not silence_segments:
        ordered = []
        for sentence in sentences:
            t0 = float(sentence.get("t0", 0.0))
            t1 = float(sentence.get("t1", 0.0))
            if t1 <= t0:
                continue
            if (t1 - t0) < min_duration:
                continue
            ordered.append({"t0": t0, "t1": t1})
        return ordered
    ordered = []
    for sentence in sentences:
        s0 = float(sentence.get("t0", 0.0))
        s1 = float(sentence.get("t1", 0.0))
        if s1 <= s0:
            continue
        for seg in silence_segments:
            t0 = max(s0, float(seg["t0"]))
            t1 = min(s1, float(seg["t1"]))
            if t1 > t0 and (t1 - t0) >= min_duration:
                ordered.append({"t0": t0, "t1": t1})
    return ordered
