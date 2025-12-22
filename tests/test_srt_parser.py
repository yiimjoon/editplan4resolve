from pathlib import Path

from VideoForge.core.srt_parser import parse_srt
from VideoForge.core.transcriber import align_sentences_to_srt


def test_parse_srt_and_align(tmp_path: Path):
    srt = tmp_path / "sample.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nHello world\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nSecond line\n",
        encoding="utf-8",
    )
    entries = parse_srt(str(srt))
    assert len(entries) == 2
    sentences = [
        {"t0": 1.1, "t1": 1.9, "text": "Hello", "metadata": {}},
        {"t0": 3.1, "t1": 3.9, "text": "Second", "metadata": {}},
    ]
    aligned = align_sentences_to_srt(sentences, entries)
    assert aligned[0]["metadata"]["script_index"] == "1"
    assert aligned[1]["metadata"]["script_index"] == "2"
