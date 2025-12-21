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
            media_start = _time_to_frames(clip["media_start"], fps)
            timeline_start = _time_to_frames(clip["timeline_start"], fps)
            duration = _time_to_frames(clip["duration"], fps)
            media_pool.InsertClips([media_item], track_type, track_index, timeline_start)

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
