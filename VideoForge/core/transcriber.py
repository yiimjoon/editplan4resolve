import logging
from bisect import bisect_left
import os
from pathlib import Path
import tempfile
import subprocess
import time
from typing import Dict, List

from VideoForge.core.srt_parser import SrtEntry

_WHISPER_MODEL = None
_WHISPER_MODEL_NAME = None
_WHISPER_MODEL_DEVICE = None

def transcribe_segments(
    video_path: str,
    segments: List[Dict[str, float]],
    model_name: str,
    task: str = "transcribe",
    language: str | None = None,
    clip_range: tuple[float, float] | None = None,
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
        desired_device = "cpu" if force_cpu else "auto"
        global _WHISPER_MODEL, _WHISPER_MODEL_NAME, _WHISPER_MODEL_DEVICE
        if (
            _WHISPER_MODEL is not None
            and _WHISPER_MODEL_NAME == model_name
            and _WHISPER_MODEL_DEVICE == desired_device
        ):
            model = _WHISPER_MODEL
        elif force_cpu:
            logger.info(">>> Loading Whisper model on CPU (device=cpu, compute_type=int8)")
            logger.info(">>> This may take 30-120 seconds on first run...")
            model = WhisperModel(model_name, device="cpu", compute_type="int8")
            logger.info(">>> CPU model loaded in %.1f seconds", time.time() - model_load_start)
            _WHISPER_MODEL = model
            _WHISPER_MODEL_NAME = model_name
            _WHISPER_MODEL_DEVICE = desired_device
        else:
            try:
                logger.info(">>> Loading Whisper model on GPU (device=auto, compute_type=int8)")
                logger.info(">>> This may take 30-120 seconds on first run...")
                model = WhisperModel(model_name, device="auto", compute_type="int8")
                logger.info(">>> GPU model loaded in %.1f seconds", time.time() - model_load_start)
                _WHISPER_MODEL = model
                _WHISPER_MODEL_NAME = model_name
                _WHISPER_MODEL_DEVICE = desired_device
            except Exception as exc:
                logger.warning(">>> GPU init failed (%s), falling back to CPU", exc)
                logger.info(">>> Loading Whisper model on CPU (fallback)")
                model = WhisperModel(model_name, device="cpu", compute_type="int8")
                logger.info(">>> CPU model loaded in %.1f seconds", time.time() - model_load_start)
                _WHISPER_MODEL = model
                _WHISPER_MODEL_NAME = model_name
                _WHISPER_MODEL_DEVICE = "cpu"

        logger.info("Transcribing audio... (this may take 30-60 seconds)")

        # VAD (Voice Activity Detection) parameters for better sentence segmentation
        # - condition_on_previous_text=False: Each segment is independent (more accurate splits)
        # - vad_filter=True: Enable silence-based segmentation
        # - vad_parameters: Fine-tune silence detection
        #   * threshold: Lower = more aggressive splitting (0.0-1.0, default 0.5)
        #   * min_silence_duration_ms: Minimum silence gap to trigger split (default 2000ms)
        #   * speech_pad_ms: Padding around speech segments (default 400ms)
        transcribe_kwargs = {
            "word_timestamps": False,
            "task": task,
            "condition_on_previous_text": False,  # Reduce long sentence merging
            "vad_filter": True,  # Enable VAD-based segmentation
            "vad_parameters": {
                "threshold": 0.3,  # More sensitive to silence (0.5 default, lower = more splits)
                "min_silence_duration_ms": 300,  # Shorter silence gap (500ms default)
                "speech_pad_ms": 0,  # Avoid padding that can erase short silences
            },
        }

        if language and language != "auto":
            transcribe_kwargs["language"] = language
        source_path = video_path
        offset = 0.0
        temp_path = None
        sam_audio_temp: List[str] = []
        if clip_range:
            duration = _get_media_duration(video_path)
            start, end = clip_range
            if duration is not None and (end <= start or end > duration + 0.5 or start < 0.0):
                logger.warning(
                    "Clip range %.2fs - %.2fs is outside media duration (%.2fs); skipping trim.",
                    float(start),
                    float(end),
                    float(duration),
                )
                clip_range = None
            elif end > start:
                offset = float(start)
                temp_path = _extract_audio_segment(video_path, start, end)
                source_path = temp_path
                logger.info(
                    "Using trimmed audio segment: %.2fs - %.2fs",
                    float(start),
                    float(end),
                )

        use_sam_audio = False
        try:
            from VideoForge.config.config_manager import Config

            use_sam_audio = bool(Config.get("use_sam_audio_preprocessing", False))
        except Exception:
            use_sam_audio = False

        if use_sam_audio:
            audio_input = source_path
            if not str(audio_input).lower().endswith(".wav"):
                try:
                    extracted_audio = _extract_audio_full(str(audio_input))
                    sam_audio_temp.append(extracted_audio)
                    audio_input = extracted_audio
                except Exception as exc:
                    logger.warning("SAM Audio extraction failed, using original audio: %s", exc)
                    audio_input = source_path
            try:
                from VideoForge.adapters.sam_audio_adapter import SAMAudioAdapter

                sam_audio = SAMAudioAdapter()
                preprocessed = sam_audio.preprocess_audio(Path(str(audio_input)))
                if preprocessed and preprocessed.exists():
                    source_path = str(preprocessed)
                    if str(preprocessed) != str(audio_input):
                        sam_audio_temp.append(str(preprocessed))
                    logger.info("SAM Audio preprocessed: %s", preprocessed)
                else:
                    logger.warning("SAM Audio preprocessing failed, using original audio")
                    source_path = str(audio_input)
            except Exception as exc:
                logger.warning("SAM Audio preprocessing error: %s", exc)
                source_path = str(audio_input)
        results, _info = model.transcribe(source_path, **transcribe_kwargs)
        sentences: List[Dict[str, float]] = []
        segment_count = 0
        try:
            for seg in results:
                seg_id = _segment_for_time(segments, float(seg.start))
                sentences.append(
                    {
                        "t0": float(seg.start) + offset,
                        "t1": float(seg.end) + offset,
                        "text": seg.text.strip(),
                        "confidence": float(getattr(seg, "avg_logprob", 0.0)),
                        "segment_id": seg_id,
                        "metadata": {},
                    }
                )
                segment_count += 1
                if segment_count % 5 == 0:
                    elapsed = time.time() - model_load_start
                    logger.info(
                        ">>> Transcribed %d sentences... (%.1f seconds elapsed)",
                        segment_count,
                        elapsed,
                    )
        finally:
            try:
                results.close()
            except Exception:
                pass

        logger.info("Transcription complete: %s sentences", len(sentences))
        if len(sentences) > 1:
            gaps = []
            for prev, curr in zip(sentences, sentences[1:]):
                gap = float(curr["t0"]) - float(prev["t1"])
                if gap > 0:
                    gaps.append(gap)
            if gaps:
                logger.info(
                    "Detected %d silence gaps (max %.2fs, avg %.2fs)",
                    len(gaps),
                    max(gaps),
                    sum(gaps) / len(gaps),
                )
        logger.info(">>> Returning %d sentences to caller", len(sentences))
        cleanup_paths = []
        if temp_path:
            cleanup_paths.append(temp_path)
        cleanup_paths.extend(sam_audio_temp)
        for path in set(cleanup_paths):
            try:
                os.unlink(path)
            except Exception:
                pass
        return sentences
    except Exception as exc:
        logger.error("Whisper transcription failed: %s", exc, exc_info=True)
        raise RuntimeError("Whisper transcription failed") from exc


def _extract_audio_segment(video_path: str, start: float, end: float) -> str:
    """Extract a temporary WAV for the requested time range."""
    temp_fd, temp_path = tempfile.mkstemp(suffix=".wav", prefix="vf_trim_")
    os.close(temp_fd)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(max(0.0, float(start))),
        "-to",
        str(max(0.0, float(end))),
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        temp_path,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            encoding="utf-8",
            errors="ignore",
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("ffmpeg segment extraction timed out") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg segment extraction failed: {proc.stderr.strip()}")
    return temp_path


def _extract_audio_full(video_path: str) -> str:
    """Extract a temporary WAV for the full media."""
    temp_fd, temp_path = tempfile.mkstemp(suffix=".wav", prefix="vf_audio_")
    os.close(temp_fd)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        temp_path,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            encoding="utf-8",
            errors="ignore",
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("ffmpeg audio extraction timed out") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {proc.stderr.strip()}")
    return temp_path


def _get_media_duration(video_path: str) -> float | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        video_path,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip())
    except Exception:
        return None


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

    entries_sorted = sorted(srt_entries, key=lambda entry: entry.start)
    ends = [entry.end for entry in entries_sorted]
    entry_count = len(entries_sorted)

    aligned: List[Dict[str, float]] = []
    for sentence in sentences:
        s0 = float(sentence["t0"])
        s1 = float(sentence["t1"])
        best_match = None
        best_overlap = 0.0
        idx = bisect_left(ends, s0)
        while idx < entry_count and entries_sorted[idx].start < s1:
            entry = entries_sorted[idx]
            overlap = _overlap(s0, s1, entry.start, entry.end)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = entry
            idx += 1
        metadata = sentence.get("metadata") or {}
        if best_match:
            metadata = {**metadata, "script_index": str(best_match.index), "source": "srt"}
        sentence["metadata"] = metadata
        aligned.append(sentence)
    return aligned
