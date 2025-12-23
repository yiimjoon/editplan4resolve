import json
import logging
import math
import subprocess
from pathlib import Path
from typing import Dict, List


def _ffprobe_duration(video_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        video_path,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
            encoding="utf-8",
            errors="ignore",
        )
        data = json.loads(proc.stdout)
        return float(data.get("format", {}).get("duration", 0.0))
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Command timed out after 300s: {cmd[0]}") from exc
    except Exception:
        return 0.0


def _extract_pcm(video_path: str, sample_rate: int = 16000) -> bytes:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        video_path,
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=300)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Command timed out after 300s: {cmd[0]}") from exc
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))
    return proc.stdout


def extract_silence_segments(
    video_path: str,
    threshold_db: float,
    min_keep: float,
    buffer_before: float,
    buffer_after: float,
) -> List[Dict[str, float]]:
    """
    Adapter boundary for AutoResolve silence detection.
    Returns speech segments: [{"t0": float, "t1": float, "type": "speech"}].
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(video_path)

    duration = _ffprobe_duration(video_path)
    if duration <= 0:
        return []

    logger = logging.getLogger(__name__)
    try:
        pcm = _extract_pcm(video_path)
    except Exception as exc:
        logger.warning("Silence detection failed, using full duration: %s", exc)
        return [
            {
                "t0": 0.0,
                "t1": duration,
                "type": "speech",
                "metadata": {"warning": "silence_detection_failed"},
            }
        ]

    sample_rate = 16000
    window_sec = 0.1
    window_size = int(sample_rate * window_sec)
    if window_size <= 0:
        logger.warning("Invalid window size for silence detection, using full duration.")
        return [
            {
                "t0": 0.0,
                "t1": duration,
                "type": "speech",
                "metadata": {"warning": "silence_detection_failed"},
            }
        ]

    total_samples = len(pcm) // 2
    num_windows = max(1, total_samples // window_size)

    speech_flags: List[bool] = []
    for i in range(num_windows):
        start = i * window_size * 2
        end = start + window_size * 2
        chunk = pcm[start:end]
        if not chunk:
            break
        samples = memoryview(chunk).cast("h")
        if len(samples) == 0:
            speech_flags.append(False)
            continue
        rms = math.sqrt(sum(s * s for s in samples) / len(samples))
        if rms <= 0:
            db = -90.0
        else:
            db = 20.0 * math.log10(rms / 32768.0)
        speech_flags.append(db >= threshold_db)

    segments: List[Dict[str, float]] = []
    current_start = None
    for i, is_speech in enumerate(speech_flags):
        t0 = i * window_sec
        t1 = min(duration, t0 + window_sec)
        if is_speech and current_start is None:
            current_start = t0
        if not is_speech and current_start is not None:
            seg_end = t0
            if seg_end - current_start >= min_keep:
                segments.append({"t0": max(0.0, current_start - buffer_before),
                                 "t1": min(duration, seg_end + buffer_after),
                                 "type": "speech"})
            current_start = None

    if current_start is not None:
        seg_end = duration
        if seg_end - current_start >= min_keep:
            segments.append({"t0": max(0.0, current_start - buffer_before),
                             "t1": min(duration, seg_end + buffer_after),
                             "type": "speech"})

    if not segments:
        logger.warning("No speech segments detected, using full duration.")
        return [
            {
                "t0": 0.0,
                "t1": duration,
                "type": "speech",
                "metadata": {"warning": "silence_detection_failed"},
            }
        ]

    return segments


def render_silence_removed(
    video_path: str,
    segments: List[Dict[str, float]],
    output_path: str,
    source_range: tuple[float, float] | None = None,
) -> str:
    """Render a new media file with only the keep segments concatenated."""
    if not Path(video_path).exists():
        raise FileNotFoundError(video_path)
    if not segments:
        raise RuntimeError("No keep segments provided for silence removal render.")

    logger = logging.getLogger(__name__)
    clip_start = None
    clip_end = None
    if source_range:
        clip_start = float(source_range[0])
        clip_end = float(source_range[1])
        if clip_end <= clip_start:
            raise RuntimeError("Invalid source_range for silence render.")
        logger.info(
            "Silence render using source range: %.2f - %.2f",
            clip_start,
            clip_end,
        )

    merged = _merge_segments(segments)
    if clip_start is not None and clip_end is not None:
        trimmed: List[Dict[str, float]] = []
        for seg in merged:
            t0 = max(float(seg["t0"]), clip_start)
            t1 = min(float(seg["t1"]), clip_end)
            if t1 > t0:
                trimmed.append({"t0": t0 - clip_start, "t1": t1 - clip_start})
        if not trimmed:
            raise RuntimeError("No keep segments within selected clip range.")
        merged = trimmed
    filters = []
    concat_inputs = []
    for idx, seg in enumerate(merged):
        t0 = float(seg["t0"])
        t1 = float(seg["t1"])
        filters.append(
            f"[0:v]trim=start={t0}:end={t1},setpts=PTS-STARTPTS[v{idx}]"
        )
        filters.append(
            f"[0:a]atrim=start={t0}:end={t1},asetpts=PTS-STARTPTS[a{idx}]"
        )
        concat_inputs.append(f"[v{idx}][a{idx}]")
    concat_filter = "".join(concat_inputs) + f"concat=n={len(merged)}:v=1:a=1[v][a]"
    filter_complex = ";".join(filters + [concat_filter])

    cmd = ["ffmpeg", "-y"]
    if clip_start is not None and clip_end is not None:
        cmd += ["-ss", str(clip_start), "-to", str(clip_end)]
    cmd += [
        "-i",
        video_path,
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        output_path,
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
        raise RuntimeError(f"Command timed out after 300s: {cmd[0]}") from exc
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())
    return output_path


def _merge_segments(segments: List[Dict[str, float]]) -> List[Dict[str, float]]:
    merged: List[Dict[str, float]] = []
    ordered = sorted(segments, key=lambda s: float(s["t0"]))
    for seg in ordered:
        t0 = float(seg["t0"])
        t1 = float(seg["t1"])
        if not merged:
            merged.append({"t0": t0, "t1": t1})
            continue
        last = merged[-1]
        if t0 <= float(last["t1"]):
            last["t1"] = max(float(last["t1"]), t1)
        else:
            merged.append({"t0": t0, "t1": t1})
    return merged
