import hashlib
import json
import logging
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


_FALLBACK_USED = False
_FALLBACK_WARNED = False
_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_DEVICE = None
_CLIP_TOKENIZER = None
_CLIP_LOGS_QUIETED = False
_ROOT_LOG_FILTERED = False


def _seed_from_text(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:8], 16)


def _deterministic_vector(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.normal(0, 1, size=(dim,)).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def encode_text_clip(query: str) -> np.ndarray:
    """
    Adapter boundary for text embeddings. Uses OpenCLIP if available, otherwise
    deterministic vectors for tests.
    """
    logger = logging.getLogger(__name__)
    try:
        import torch  # type: ignore

        model, _preprocess, device = _load_clip_model()
        tokenizer = _load_clip_tokenizer()
        tokens = tokenizer([query])
        if hasattr(tokens, "to"):
            tokens = tokens.to(device)
        with np.errstate(all="ignore"):
            with torch.no_grad():
                text_features = model.encode_text(tokens)
        text_vec = text_features.detach().cpu().numpy().astype(np.float32)
        text_vec = text_vec[0]
        text_vec /= max(np.linalg.norm(text_vec), 1e-8)
        return text_vec
    except Exception as exc:
        _mark_fallback(logger, "CLIP model unavailable, using deterministic fallback", exc)
        return _deterministic_vector(_seed_from_text(query))


def consume_fallback_used() -> bool:
    global _FALLBACK_USED
    used = _FALLBACK_USED
    _FALLBACK_USED = False
    return used


def _mark_fallback(logger: logging.Logger, message: str, exc: Exception) -> None:
    global _FALLBACK_USED, _FALLBACK_WARNED
    _FALLBACK_USED = True
    if not _FALLBACK_WARNED:
        logger.warning("%s: %s", message, exc)
        _FALLBACK_WARNED = True
    else:
        logger.debug("%s: %s", message, exc)


def _load_clip_model(force_cpu: bool = False):
    global _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE
    if _CLIP_MODEL is not None and (not force_cpu or _CLIP_DEVICE == "cpu"):
        return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE
    _quiet_external_loggers()
    import torch  # type: ignore
    import open_clip  # type: ignore

    device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model.eval()
    model.to(device)
    _CLIP_MODEL = model
    _CLIP_PREPROCESS = preprocess
    _CLIP_DEVICE = device
    return model, preprocess, device


def _load_clip_tokenizer():
    global _CLIP_TOKENIZER
    if _CLIP_TOKENIZER is not None:
        return _CLIP_TOKENIZER
    import open_clip  # type: ignore

    _CLIP_TOKENIZER = open_clip.get_tokenizer("ViT-B-32")
    return _CLIP_TOKENIZER


def _quiet_external_loggers() -> None:
    global _CLIP_LOGS_QUIETED, _ROOT_LOG_FILTERED
    if _CLIP_LOGS_QUIETED:
        return
    for name in (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "open_clip",
        "timm",
        "urllib3",
        "PIL",
        "PIL.PngImagePlugin",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
    if not _ROOT_LOG_FILTERED:
        logging.getLogger().addFilter(_OpenClipRootFilter())
        _ROOT_LOG_FILTERED = True
    _CLIP_LOGS_QUIETED = True


class _OpenClipRootFilter(logging.Filter):
    _DROP_SNIPPETS = (
        "Parsing model identifier",
        "Loaded built-in ViT-B-32 model config",
        "Instantiating model architecture: CLIP",
        "Loading full pretrained weights from:",
        "Final image preprocessing configuration set:",
        "Model ViT-B-32 creation process complete.",
        "Parsing tokenizer identifier",
        "Attempting to load config from built-in:",
        "Using default SimpleTokenizer.",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "root":
            return True
        message = record.getMessage()
        return not any(snippet in message for snippet in self._DROP_SNIPPETS)


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


def _extract_frames(video_path: str, timestamps: Iterable[float]) -> list:
    from PIL import Image

    ts_list = [float(ts) for ts in timestamps if ts is not None]
    if not ts_list:
        return []

    fps = _ffprobe_fps(video_path)
    if fps <= 0:
        return _extract_frames_single(video_path, ts_list)

    frame_indices = sorted(
        {max(0, int(round(ts * fps))) for ts in ts_list if ts >= 0.0}
    )
    if not frame_indices:
        return _extract_frames_single(video_path, ts_list)

    expr = "+".join(f"eq(n\\,{idx})" for idx in frame_indices)
    frames: list = []
    with tempfile.TemporaryDirectory(prefix="vf_frames_") as tmpdir:
        output_pattern = str(Path(tmpdir) / "vf_frame_%03d.png")
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            video_path,
            "-vf",
            f"select={expr}",
            "-vsync",
            "0",
            "-frames:v",
            str(len(frame_indices)),
            output_pattern,
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=300)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Command timed out after 300s: {cmd[0]}") from exc
        except Exception:
            return _extract_frames_single(video_path, ts_list)

        for image_path in sorted(Path(tmpdir).glob("vf_frame_*.png")):
            try:
                frames.append(Image.open(image_path).convert("RGB"))
            except Exception:
                continue

    if not frames:
        return _extract_frames_single(video_path, ts_list)
    return frames


def _extract_frames_single(video_path: str, timestamps: Iterable[float]) -> list:
    from PIL import Image

    frames = []
    for ts in timestamps:
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{max(0.0, ts):.3f}",
            "-i",
            video_path,
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, check=True, timeout=300)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Command timed out after 300s: {cmd[0]}") from exc
        except Exception:
            continue
        if not proc.stdout:
            continue
        try:
            frames.append(Image.open(BytesIO(proc.stdout)).convert("RGB"))
        except Exception:
            continue
    return frames


def _ffprobe_fps(video_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
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
        return 0.0
    if proc.returncode != 0:
        return 0.0
    raw = proc.stdout.strip()
    if not raw:
        return 0.0
    try:
        if "/" in raw:
            num, den = raw.split("/", 1)
            den_val = float(den)
            if den_val == 0:
                return 0.0
            return float(num) / den_val
        return float(raw)
    except Exception:
        return 0.0


def encode_video_clip(
    video_path: str,
    seed_hint: Optional[str] = None,
    sample_points: Optional[Iterable[float]] = None,
) -> np.ndarray:
    """
    Adapter boundary for video embeddings. Uses OpenCLIP if available.
    """
    logger = logging.getLogger(__name__)
    path = Path(video_path)
    if seed_hint is None:
        seed_hint = f"{path.name}:{path.stat().st_size if path.exists() else 0}"
    try:
        return _encode_video_with_clip(path, sample_points, force_cpu=False)
    except Exception as exc:
        logger.warning("Video CLIP primary failed, retrying on CPU: %s", exc)
        try:
            return _encode_video_with_clip(path, sample_points, force_cpu=True, batch_size=1)
        except Exception as retry_exc:
            _mark_fallback(
                logger,
                "Video CLIP unavailable, using deterministic fallback",
                retry_exc,
            )
            return _deterministic_vector(_seed_from_text(seed_hint))


def encode_image_clip(image_path: str, seed_hint: Optional[str] = None) -> np.ndarray:
    """
    Adapter boundary for image embeddings. Uses OpenCLIP if available.
    """
    logger = logging.getLogger(__name__)
    path = Path(image_path)
    if seed_hint is None:
        seed_hint = f"{path.name}:{path.stat().st_size if path.exists() else 0}"
    try:
        model, preprocess, device = _load_clip_model()
        from PIL import Image
        import torch  # type: ignore

        image = Image.open(path).convert("RGB")
        with torch.no_grad():
            inputs = preprocess(image).unsqueeze(0).to(device)
            image_features = model.encode_image(inputs)
            image_vec = image_features.detach().cpu().numpy().astype(np.float32)[0]
        norm = np.linalg.norm(image_vec)
        if norm > 0:
            image_vec /= norm
        return image_vec
    except Exception as exc:
        logger.warning("Image CLIP primary failed, retrying on CPU: %s", exc)
        try:
            model, preprocess, device = _load_clip_model(force_cpu=True)
            from PIL import Image
            import torch  # type: ignore

            image = Image.open(path).convert("RGB")
            with torch.no_grad():
                inputs = preprocess(image).unsqueeze(0).to(device)
                image_features = model.encode_image(inputs)
                image_vec = image_features.detach().cpu().numpy().astype(np.float32)[0]
            norm = np.linalg.norm(image_vec)
            if norm > 0:
                image_vec /= norm
            return image_vec
        except Exception as retry_exc:
            _mark_fallback(
                logger,
                "Image CLIP unavailable, using deterministic fallback",
                retry_exc,
            )
            return _deterministic_vector(_seed_from_text(seed_hint))


def _encode_video_with_clip(
    path: Path,
    sample_points: Optional[Iterable[float]],
    force_cpu: bool,
    batch_size: int = 6,
) -> np.ndarray:
    model, preprocess, device = _load_clip_model(force_cpu=force_cpu)
    duration = _ffprobe_duration(str(path))
    if duration <= 0:
        raise RuntimeError("ffprobe duration unavailable")
    if sample_points is None:
        from VideoForge.broll.indexer import smart_frame_sampling

        sample_points = smart_frame_sampling(duration, max_frames=12, base_frames=6)
    frames = _extract_frames(str(path), sample_points)
    if not frames:
        raise RuntimeError("Failed to extract frames")
    import torch  # type: ignore

    image_vecs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            inputs = torch.stack([preprocess(frame) for frame in batch]).to(device)
            image_features = model.encode_image(inputs)
            image_vecs.append(image_features.detach().cpu().numpy().astype(np.float32))
    all_vecs = np.concatenate(image_vecs, axis=0)
    mean_vec = np.mean(all_vecs, axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec /= norm
    return mean_vec
