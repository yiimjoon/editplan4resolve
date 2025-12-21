from typing import Dict, List

from VideoForge.adapters.audio_adapter import extract_silence_segments


def process_silence(
    video_path: str,
    threshold_db: float,
    min_keep: float,
    buffer_before: float,
    buffer_after: float,
) -> List[Dict[str, float]]:
    """
    Returns speech segments for the main video.
    """
    return extract_silence_segments(
        video_path=video_path,
        threshold_db=threshold_db,
        min_keep=min_keep,
        buffer_before=buffer_before,
        buffer_after=buffer_after,
    )
