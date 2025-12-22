import numpy as np

from VideoForge.adapters import embedding_adapter


def test_encode_video_clip_fallback_is_deterministic(monkeypatch):
    def _fail_load():
        raise RuntimeError("no clip")

    monkeypatch.setattr(embedding_adapter, "_load_clip_model", _fail_load)
    vec1 = embedding_adapter.encode_video_clip("missing.mp4", seed_hint="seed")
    vec2 = embedding_adapter.encode_video_clip("missing.mp4", seed_hint="seed")
    assert vec1.shape == vec2.shape
    assert np.allclose(vec1, vec2)
    assert abs(np.linalg.norm(vec1) - 1.0) < 1e-6
    assert embedding_adapter.consume_fallback_used() is True
