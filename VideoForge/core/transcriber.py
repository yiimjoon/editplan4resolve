from typing import Dict, List


def transcribe_segments(
    video_path: str,
    segments: List[Dict[str, float]],
    model_name: str,
) -> List[Dict[str, float]]:
    """
    Transcribe audio and return sentence-level segments.
    Each sentence: {"t0": float, "t1": float, "text": str, "confidence": float, "segment_id": int}
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore

        model = WhisperModel(model_name, device="auto", compute_type="int8")
        results, _info = model.transcribe(video_path, word_timestamps=False)
        sentences: List[Dict[str, float]] = []
        for seg in results:
            seg_id = _segment_for_time(segments, float(seg.start))
            sentences.append(
                {
                    "t0": float(seg.start),
                    "t1": float(seg.end),
                    "text": seg.text.strip(),
                    "confidence": float(getattr(seg, "avg_logprob", 0.0)),
                    "segment_id": seg_id,
                }
            )
        return sentences
    except Exception:
        return []


def _segment_for_time(segments: List[Dict[str, float]], t0: float) -> int | None:
    for seg in segments:
        if float(seg["t0"]) <= t0 <= float(seg["t1"]):
            return int(seg.get("id")) if seg.get("id") is not None else None
    return None
