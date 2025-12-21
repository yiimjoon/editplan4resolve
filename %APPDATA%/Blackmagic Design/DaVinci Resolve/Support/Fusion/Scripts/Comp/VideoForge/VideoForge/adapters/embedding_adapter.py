import hashlib
from pathlib import Path
from typing import Optional

import numpy as np


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
    except Exception:
        return _deterministic_vector(_seed_from_text(query))


def encode_video_clip(video_path: str, seed_hint: Optional[str] = None) -> np.ndarray:
    """
    Adapter boundary for video embeddings. Uses deterministic vectors for MVP.
    """
    path = Path(video_path)
    if seed_hint is None:
        seed_hint = f"{path.name}:{path.stat().st_size if path.exists() else 0}"
    return _deterministic_vector(_seed_from_text(seed_hint))
