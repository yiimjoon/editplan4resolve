import json
import os
import sys
from pathlib import Path
from typing import Dict, List


def _add_resolve_script_paths() -> None:
    candidates = []
    env_root = os.environ.get("RESOLVE_SCRIPT_API")
    if env_root:
        candidates.append(Path(env_root))

    if os.name == "nt":
        candidates.extend(
            [
                Path(os.environ.get("PROGRAMDATA", ""))
                / "Blackmagic Design"
                / "DaVinci Resolve"
                / "Support"
                / "Developer"
                / "Scripting"
                / "Modules",
                Path(os.environ.get("APPDATA", ""))
                / "Blackmagic Design"
                / "DaVinci Resolve"
                / "Support"
                / "Developer"
                / "Scripting"
                / "Modules",
                Path("C:/Program Files/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules"),
                Path("C:/Program Files/Blackmagic Design/DaVinci Resolve/Support/Developer/Scripting/Modules"),
                Path("C:/Program Files/Blackmagic Design/DaVinci Resolve/RESOLVE_SCRIPT_API/Modules"),
            ]
        )
    elif sys.platform == "darwin":
        candidates.extend(
            [
                Path(
                    "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules"
                ),
                Path(
                    "/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Resources/Developer/Scripting/Modules"
                ),
            ]
        )
    else:
        candidates.extend(
            [
                Path("/opt/resolve/Developer/Scripting/Modules"),
                Path("/opt/resolve/Resolve.app/Contents/Resources/Developer/Scripting/Modules"),
            ]
        )

    for path in candidates:
        if path and path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            break


def _get_resolve():
    _add_resolve_script_paths()
    try:
        import DaVinciResolveScript as dvr_script  # type: ignore

        return dvr_script.scriptapp("Resolve")
    except Exception as exc:
        raise RuntimeError("DaVinciResolveScript not available") from exc


def _ensure_media(media_pool, path: str):
    existing = media_pool.GetRootFolder().GetClipList()
    for clip in existing:
        if clip.GetClipProperty("File Path") == path:
            return clip
    imported = media_pool.ImportMedia([path])
    return imported[0] if imported else None


def _time_to_frames(seconds: float, fps: float) -> int:
    return int(round(seconds * fps))


def _insert_clip(
    media_pool,
    media_item,
    track_type: str,
    track_index: int,
    timeline_start: int,
    media_start: int,
    media_end: int,
) -> None:
    append_method = getattr(media_pool, "AppendToTimeline", None)
    if callable(append_method):
        payload = {
            "mediaPoolItem": media_item,
            "startFrame": int(media_start),
            "endFrame": int(media_end),
            "recordFrame": int(timeline_start),
            "trackIndex": int(track_index),
            "trackType": track_type,
        }
        try:
            append_method([payload])
            return
        except Exception:
            pass
    media_pool.InsertClips([media_item], track_type, int(track_index), int(timeline_start))


def export_to_resolve(timeline_json: Dict, timeline_name_prefix: str) -> None:
    resolve = _get_resolve()
    project_manager = resolve.GetProjectManager()
    project = project_manager.GetCurrentProject()
    if project is None:
        project = project_manager.CreateProject("VideoForge")

    media_pool = project.GetMediaPool()
    timeline_name = f"{timeline_name_prefix}{timeline_json['metadata']['project_id']}"
    timeline = project.GetTimelineByName(timeline_name)
    if timeline is None:
        timeline = media_pool.CreateEmptyTimeline(timeline_name)

    fps = float(timeline_json["settings"]["fps"] or 24.0)

    for track in timeline_json["tracks"]:
        track_type = "video"
        track_index = int(track["index"])
        for clip in track["clips"]:
            media_item = _ensure_media(media_pool, clip["file"])
            if media_item is None:
                continue
            media_start = _time_to_frames(float(clip.get("media_start", 0.0)), fps)
            timeline_start = _time_to_frames(float(clip.get("timeline_start", 0.0)), fps)
            media_end_sec = clip.get("media_end")
            if media_end_sec is None:
                media_end_sec = float(clip.get("media_start", 0.0)) + float(
                    clip.get("duration", 0.0)
                )
            media_end = _time_to_frames(float(media_end_sec), fps)
            if media_end <= media_start:
                media_end = media_start + 1
            _insert_clip(
                media_pool,
                media_item,
                track_type,
                track_index,
                timeline_start,
                media_start,
                media_end,
            )

            if track["type"] == "video_broll" and not clip.get("audio_enabled", False):
                items = timeline.GetItemListInTrack(track_type, track_index)
                if items:
                    last_item = items[-1]
                    try:
                        last_item.SetProperty("Audio Enabled", False)
                    except Exception:
                        pass

    project.SetCurrentTimeline(timeline)


def export_timeline_file(timeline_json_path: str, timeline_name_prefix: str) -> None:
    data = json.loads(Path(timeline_json_path).read_text(encoding="utf-8"))
    export_to_resolve(data, timeline_name_prefix)
