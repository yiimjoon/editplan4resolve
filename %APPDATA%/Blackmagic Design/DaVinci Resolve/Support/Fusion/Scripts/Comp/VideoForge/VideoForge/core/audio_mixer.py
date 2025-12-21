from typing import Dict, List


def mute_broll_audio(clips: List[Dict]) -> List[Dict]:
    for clip in clips:
        clip["audio_enabled"] = False
    return clips
