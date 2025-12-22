import hashlib
import json
import logging
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


_FALLBACK_USED = False
_FALLBACK_WARNED = False
_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_DEVICE = None


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
        import open_clip  # type: ignore

        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        tokens = tokenizer([query])
        with np.errstate(all="ignore"):
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


def _load_clip_model():
    global _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE
    import torch  # type: ignore
    import open_clip  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model.eval()
    model.to(device)
    _CLIP_MODEL = model
    _CLIP_PREPROCESS = preprocess
    _CLIP_DEVICE = device
    return model, preprocess, device


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
        model, preprocess, device = _load_clip_model()
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

        with torch.no_grad():
            inputs = torch.stack([preprocess(frame) for frame in frames]).to(device)
            image_features = model.encode_image(inputs)
            image_vecs = image_features.detach().cpu().numpy().astype(np.float32)
        mean_vec = np.mean(image_vecs, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec /= norm
        return mean_vec
    except Exception as exc:
        _mark_fallback(logger, "Video CLIP unavailable, using deterministic fallback", exc)
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
        _mark_fallback(logger, "Image CLIP unavailable, using deterministic fallback", exc)
        return _deterministic_vector(_seed_from_text(seed_hint))
