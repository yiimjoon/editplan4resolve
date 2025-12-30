"""Tool registry and core Resolve tools for VideoForge agents."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from functools import wraps
from typing import Any, Callable, Dict, Optional

from VideoForge.integrations.resolve_api import ResolveAPI
from VideoForge.core.sync_matcher import SyncMatcher
from VideoForge.broll.matcher import BrollMatcher
from VideoForge.cli.main import load_settings

logger = logging.getLogger(__name__)

ToolCallable = Callable[..., Any]
TOOL_REGISTRY: Dict[str, ToolCallable] = {}


def videoforge_tool(name: str, description: str, parameters: Dict[str, Any]) -> Callable[[ToolCallable], ToolCallable]:
    """Decorator to register a function as a VideoForge agent tool."""

    def decorator(func: ToolCallable) -> ToolCallable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        wrapper._tool_metadata = {
            "name": name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": parameters,
                "required": list(parameters.keys()),
            },
        }
        TOOL_REGISTRY[name] = wrapper
        return wrapper

    return decorator


def get_tool_schema(name: str) -> Optional[Dict[str, Any]]:
    """Return the JSON schema metadata for a tool name."""
    func = TOOL_REGISTRY.get(name)
    if func is None:
        return None
    return getattr(func, "_tool_metadata", None)


def _require_main_thread(tool_name: str) -> Optional[Dict[str, Any]]:
    if threading.current_thread() is threading.main_thread():
        return None
    return {
        "success": False,
        "error": f"Resolve API must run on main thread (tool={tool_name}).",
        "suggestion": "Run this action from the main UI thread.",
    }


@videoforge_tool(
    name="get_timeline_info",
    description="Return basic metadata about the current timeline.",
    parameters={},
)
def get_timeline_info() -> Dict[str, Any]:
    error = _require_main_thread("get_timeline_info")
    if error:
        return error
    api = ResolveAPI()
    timeline = api.get_current_timeline()
    fps = api.get_timeline_fps()
    name = None
    get_name = getattr(timeline, "GetName", None)
    if callable(get_name):
        try:
            name = get_name()
        except Exception:
            name = None
    get_timecode = getattr(timeline, "GetCurrentTimecode", None)
    timecode = None
    if callable(get_timecode):
        try:
            timecode = get_timecode()
        except Exception:
            timecode = None
    return {
        "success": True,
        "name": name or "Timeline",
        "fps": fps,
        "timecode": timecode,
        "video_tracks": int(timeline.GetTrackCount("video") or 0),
        "audio_tracks": int(timeline.GetTrackCount("audio") or 0),
    }


@videoforge_tool(
    name="insert_clip",
    description="Insert a media clip into the current timeline.",
    parameters={
        "path": {"type": "string"},
        "track_type": {"type": "string"},
        "track_index": {"type": "integer"},
        "start_frame": {"type": "integer"},
        "start_seconds": {"type": "number"},
        "start_timecode": {"type": "string"},
    },
)
def insert_clip(
    path: str,
    track_type: str = "video",
    track_index: int = 1,
    start_frame: Optional[int] = None,
    start_seconds: Optional[float] = None,
    start_timecode: Optional[str] = None,
) -> Dict[str, Any]:
    error = _require_main_thread("insert_clip")
    if error:
        return error
    api = ResolveAPI()
    timeline = api.get_current_timeline()
    fps = api.get_timeline_fps()
    track_type = (track_type or "video").lower()
    if track_type not in {"video", "audio"}:
        raise ValueError("track_type must be 'video' or 'audio'.")

    timeline_start = api.get_timeline_start_frame()
    if start_timecode:
        tc_frame = api._timecode_to_frames(str(start_timecode).strip(), fps)
        if tc_frame is not None:
            start_frame = int(tc_frame)
    if start_frame is None and start_seconds is not None:
        try:
            start_frame = int(round(float(start_seconds) * float(fps))) + int(timeline_start)
        except Exception:
            start_frame = None
    if start_frame is None:
        get_frame = getattr(timeline, "GetCurrentFrame", None)
        if callable(get_frame):
            try:
                start_frame = int(get_frame())
            except Exception:
                start_frame = None
        if start_frame is None:
            start_frame = api._get_playhead_frame(timeline) or 0

    media_item = None
    path_str = str(path).strip()
    path_obj = Path(path_str)
    if path_obj.exists():
        media_item = api.import_media_if_needed(str(path_obj))
    if media_item is None:
        media_item = _find_media_pool_item_by_name(api.get_media_pool(), path_obj.name)
    if media_item is None:
        raise RuntimeError(
            f"Failed to import media: {path}. "
            "Use an absolute path or ensure the clip is in the media pool."
        )

    inserted = api.insert_clip_at_position(
        track_type, int(track_index), media_item, int(start_frame)
    )
    actual_start = api.get_item_start_frame(inserted) if inserted else None
    actual_track_index = ResolveAPI._safe_call(inserted, "GetTrackIndex") if inserted else None
    actual_track_type = ResolveAPI._safe_call(inserted, "GetTrackType") if inserted else None
    if inserted is not None and actual_start is not None:
        if abs(int(actual_start) - int(start_frame)) > 2:
            if api._set_item_start_frame(inserted, int(start_frame)):
                actual_start = api.get_item_start_frame(inserted)
    return {
        "success": inserted is not None,
        "start_frame": int(start_frame),
        "actual_start_frame": int(actual_start) if actual_start is not None else None,
        "actual_timecode": api._frames_to_timecode(int(actual_start), fps)
        if actual_start is not None
        else None,
        "actual_track_index": actual_track_index,
        "actual_track_type": actual_track_type,
        "timeline_start_frame": int(timeline_start),
    }


@videoforge_tool(
    name="delete_clip_at_playhead",
    description="Delete the timeline clip under the playhead.",
    parameters={},
)
def delete_clip_at_playhead() -> Dict[str, Any]:
    error = _require_main_thread("delete_clip_at_playhead")
    if error:
        return error
    api = ResolveAPI()
    timeline = api.get_current_timeline()
    items = api.get_items_at_playhead(["video", "audio"])
    if not items:
        clip = api.get_clip_at_playhead()
        if clip is None:
            return {"success": False, "error": "No clip at playhead"}
        items = [clip]

    delete_candidates = [
        ("DeleteClips", timeline, items),
        ("DeleteItems", timeline, items),
        ("RemoveItems", timeline, items),
    ]
    deleted = False
    for name, obj, payload in delete_candidates:
        method = getattr(obj, name, None)
        if not callable(method):
            continue
        try:
            result = method(payload)
            logger.info("Agent tool delete: %s returned %s", name, result)
            deleted = True
            break
        except Exception as exc:
            logger.debug("Agent tool delete: %s failed: %s", name, exc)

    if not deleted:
        for item in items:
            per_item_candidates = [
                ("DeleteClip", timeline, item),
                ("RemoveClip", timeline, item),
                ("Delete", item, None),
                ("Remove", item, None),
            ]
            for name, obj, payload in per_item_candidates:
                method = getattr(obj, name, None)
                if not callable(method):
                    continue
                try:
                    if payload is None:
                        result = method()
                    else:
                        result = method(payload)
                    logger.info("Agent tool delete: %s returned %s", name, result)
                    deleted = True
                    break
                except Exception as exc:
                    logger.debug("Agent tool delete: %s failed: %s", name, exc)
            if deleted:
                continue

    if not deleted:
        return {"success": False, "error": "Delete method not available"}
    return {"success": True, "deleted_items": len(items)}


def _find_media_pool_item_by_name(media_pool: Any, name: str) -> Optional[Any]:
    try:
        root = media_pool.GetRootFolder()
    except Exception:
        return None
    target = str(name or "").strip().lower()
    if not target:
        return None
    for clip in root.GetClipList() or []:
        try:
            props = clip.GetClipProperty()
        except Exception:
            props = {}
        file_path = str(props.get("File Path") or "").strip()
        clip_name = str(props.get("Clip Name") or props.get("File Name") or "").strip()
        if file_path and Path(file_path).name.lower() == target:
            return clip
        if clip_name and clip_name.lower() == target:
            return clip
    return None


@videoforge_tool(
    name="get_clips_at_playhead",
    description="Return clips that overlap the playhead.",
    parameters={
        "track_types": {"type": "array", "items": {"type": "string"}},
    },
)
def get_clips_at_playhead(track_types: Optional[list[str]] = None) -> Dict[str, Any]:
    error = _require_main_thread("get_clips_at_playhead")
    if error:
        return error
    api = ResolveAPI()
    items = api.get_items_at_playhead(track_types or ["video"])
    results = []
    for item in items:
        start, end = api._get_item_range(item)
        media_path = api.get_item_media_path(item)
        name = ResolveAPI._safe_call(item, "GetName")
        if not name and media_path:
            name = str(media_path).split("\\")[-1]
        results.append(
            {
                "name": name or "Clip",
                "track_index": ResolveAPI._safe_call(item, "GetTrackIndex"),
                "track_type": ResolveAPI._safe_call(item, "GetTrackType"),
                "start_frame": start,
                "end_frame": end,
                "source_path": media_path,
            }
        )
    return {"success": True, "clips": results}


@videoforge_tool(
    name="sync_clips_from_timeline",
    description="Sync clips using audio patterns and place them on the timeline.",
    parameters={
        "reference_path": {"type": "string"},
        "target_paths": {"type": "array", "items": {"type": "string"}},
        "mode": {"type": "string"},
    },
)
def sync_clips_from_timeline(
    reference_path: str,
    target_paths: list[str],
    mode: str = "same",
) -> Dict[str, Any]:
    error = _require_main_thread("sync_clips_from_timeline")
    if error:
        return error
    if not reference_path or not target_paths:
        return {"success": False, "error": "Reference and target paths are required."}
    matcher = SyncMatcher()
    results = matcher.sync_multiple(
        reference_video=Path(reference_path),
        target_videos=[Path(path) for path in target_paths],
        mode=mode,
    )
    api = ResolveAPI()
    placed = api.add_synced_clips_to_timeline(
        reference_path=reference_path,
        sync_results=results,
        start_frame=0,
    )
    serializable = {str(path): data for path, data in results.items()}
    return {"success": bool(placed), "results": serializable}


@videoforge_tool(
    name="match_broll_for_segment",
    description="Find candidate B-roll clips for a text segment.",
    parameters={
        "query": {"type": "string"},
        "duration": {"type": "number"},
        "limit": {"type": "integer"},
    },
)
def match_broll_for_segment(query: str, duration: float = 3.0, limit: int = 5) -> Dict[str, Any]:
    if not query:
        return {"success": False, "error": "Query text is required."}
    settings = load_settings()
    matcher = BrollMatcher(db=None, settings=settings)
    sentence = {"id": 1, "text": query, "t0": 0.0, "t1": float(duration)}
    matches = matcher.match([sentence])
    limited = matches[: max(1, int(limit))]
    return {"success": True, "matches": limited}

