from pathlib import Path

import pytest

from VideoForge.adapters.sam3_adapter import SAM3Adapter
from VideoForge.adapters.sam_audio_adapter import SAMAudioAdapter


def test_sam3_video_tagging() -> None:
    test_video = Path("test_data/sample_person.mp4")
    if not test_video.exists():
        pytest.skip("Test video not found")
    adapter = SAM3Adapter()
    tags = adapter.generate_vision_tags(test_video, prompts=["person"], max_frames=30)
    assert "person" in tags


def test_sam_audio_preprocessing() -> None:
    test_audio = Path("test_data/sample_speech.wav")
    if not test_audio.exists():
        pytest.skip("Test audio not found")
    adapter = SAMAudioAdapter()
    output_path = adapter.preprocess_audio(test_audio)
    assert output_path is not None
    assert output_path.exists()
