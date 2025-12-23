from __future__ import annotations

from VideoForge.adapters.audio_adapter import build_silence_render_command


def test_build_silence_render_command_includes_trim_and_concat() -> None:
    cmd = build_silence_render_command(
        "input.mp4",
        [{"t0": 0.0, "t1": 1.0}, {"t0": 2.0, "t1": 3.0}],
        "out.mp4",
        source_range=(10.0, 20.0),
    )
    cmd_str = " ".join(cmd)
    assert "-ss 10.0" in cmd_str
    assert "-to 20.0" in cmd_str
    assert "trim=start=0.0:end=1.0" in cmd_str
    assert "concat=n=2" in cmd_str
