from __future__ import annotations

from VideoForge.core.transcript_utils import split_sentence, wrap_text


def test_wrap_text_basic() -> None:
    lines = wrap_text("hello world", max_chars=50)
    assert lines == ["hello world"]


def test_split_sentence_preserves_duration() -> None:
    sentence = {"t0": 0.0, "t1": 10.0, "text": "this is a longer sentence that should split"}
    parts = split_sentence(sentence, max_chars=10, gap_sec=0.2)
    assert len(parts) > 1
    assert abs(parts[0]["t0"] - 0.0) < 1e-6
    assert abs(parts[-1]["t1"] - 10.0) < 1e-6
    for part in parts:
        assert part["t1"] >= part["t0"]
