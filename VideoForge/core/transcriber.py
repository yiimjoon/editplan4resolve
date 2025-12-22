import logging
import os
import time
from typing import Dict, List

from VideoForge.core.srt_parser import SrtEntry

def transcribe_segments(
    video_path: str,
    segments: List[Dict[str, float]],
    model_name: str,
    task: str = "transcribe",
    language: str | None = None,
) -> List[Dict[str, float]]:
    """
    Transcribe audio and return sentence-level segments.
    Each sentence: {"t0": float, "t1": float, "text": str, "confidence": float, "segment_id": int}
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("=== WHISPER TRANSCRIPTION START ===")
    logger.info("Video: %s", os.path.basename(video_path))
    logger.info("Model: %s, Task: %s, Language: %s", model_name, task, language or "auto")
    logger.info("=" * 60)
    try:
        from faster_whisper import WhisperModel  # type: ignore

        model_load_start = time.time()
        force_cpu = (
            os.environ.get("VIDEOFORGE_FORCE_CPU") == "1"
            or os.environ.get("CT2_USE_CUDA") == "0"
            or os.environ.get("CUDA_VISIBLE_DEVICES") in {"", "-1"}
        )
        if force_cpu:
            logger.info(">>> Loading Whisper model on CPU (device=cpu, compute_type=int8)")
            logger.info(">>> This may take 30-120 seconds on first run...")
            model = WhisperModel(model_name, device="cpu", compute_type="int8")
            logger.info(">>> CPU model loaded in %.1f seconds", time.time() - model_load_start)
        else:
            try:
                logger.info(">>> Loading Whisper model on GPU (device=auto, compute_type=int8)")
                logger.info(">>> This may take 30-120 seconds on first run...")
                model = WhisperModel(model_name, device="auto", compute_type="int8")
                logger.info(">>> GPU model loaded in %.1f seconds", time.time() - model_load_start)
            except Exception as exc:
                logger.warning(">>> GPU init failed (%s), falling back to CPU", exc)
                logger.info(">>> Loading Whisper model on CPU (fallback)")
                model = WhisperModel(model_name, device="cpu", compute_type="int8")
                logger.info(">>> CPU model loaded in %.1f seconds", time.time() - model_load_start)

        logger.info("Transcribing audio... (this may take 30-60 seconds)")
        transcribe_kwargs = {"word_timestamps": False, "task": task}
        if language and language != "auto":
            transcribe_kwargs["language"] = language
        results, _info = model.transcribe(video_path, **transcribe_kwargs)
        sentences: List[Dict[str, float]] = []
        segment_count = 0
        for seg in results:
            seg_id = _segment_for_time(segments, float(seg.start))
            sentences.append(
                {
                    "t0": float(seg.start),
                    "t1": float(seg.end),
                    "text": seg.text.strip(),
                    "confidence": float(getattr(seg, "avg_logprob", 0.0)),
                    "segment_id": seg_id,
                    "metadata": {},
                }
            )
            segment_count += 1
            if segment_count % 5 == 0:
                elapsed = time.time() - model_load_start
                logger.info(">>> Transcribed %d sentences... (%.1f seconds elapsed)", segment_count, elapsed)
        logger.info("Transcription complete: %s sentences", len(sentences))
        return sentences
    except Exception as exc:
        logger.error("Whisper transcription failed: %s", exc, exc_info=True)
        raise RuntimeError("Whisper transcription failed") from exc


def _segment_for_time(segments: List[Dict[str, float]], t0: float) -> int | None:
    for seg in segments:
        if float(seg["t0"]) <= t0 <= float(seg["t1"]):
            return int(seg.get("id")) if seg.get("id") is not None else None
    return None


def align_sentences_to_srt(
    sentences: List[Dict[str, float]], srt_entries: List[SrtEntry]
) -> List[Dict[str, float]]:
    if not srt_entries:
        return sentences

    def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
        return max(0.0, min(a1, b1) - max(a0, b0))

    aligned: List[Dict[str, float]] = []
    for sentence in sentences:
        best_match = None
        best_overlap = 0.0
        for entry in srt_entries:
            overlap = _overlap(
                float(sentence["t0"]), float(sentence["t1"]), entry.start, entry.end
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = entry
        metadata = sentence.get("metadata") or {}
        if best_match:
            metadata = {**metadata, "script_index": str(best_match.index), "source": "srt"}
        sentence["metadata"] = metadata
        aligned.append(sentence)
    return aligned
