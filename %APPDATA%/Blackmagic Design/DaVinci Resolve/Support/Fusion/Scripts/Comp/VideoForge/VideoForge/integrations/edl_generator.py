from typing import Dict, List


def _frames_to_tc(frames: int, fps: float) -> str:
    fps_int = int(round(fps)) or 24
    hours = frames // (3600 * fps_int)
    frames %= 3600 * fps_int
    minutes = frames // (60 * fps_int)
    frames %= 60 * fps_int
    seconds = frames // fps_int
    frame = frames % fps_int
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frame:02d}"


def _time_to_frames(seconds: float, fps: float) -> int:
    return int(round(seconds * fps))


def generate_edl(timeline_json: Dict) -> str:
    fps = float(timeline_json["settings"]["fps"] or 24.0)
    events: List[str] = []
    event_num = 1

    for track in timeline_json["tracks"]:
        for clip in sorted(track["clips"], key=lambda c: c["timeline_start"]):
            src_in = _time_to_frames(clip["media_start"], fps)
            src_out = _time_to_frames(clip["media_end"], fps)
            rec_in = _time_to_frames(clip["timeline_start"], fps)
            rec_out = _time_to_frames(clip["timeline_start"] + clip["duration"], fps)

            events.append(
                f"{event_num:03d}  AX       V     C        "
                f"{_frames_to_tc(src_in, fps)} {_frames_to_tc(src_out, fps)} "
                f"{_frames_to_tc(rec_in, fps)} {_frames_to_tc(rec_out, fps)}"
            )
            events.append(f"* FROM CLIP NAME: {clip['file']}")
            event_num += 1

    header = ["TITLE: VideoForge", "FCM: NON-DROP FRAME"]
    return "\n".join(header + events) + "\n"
