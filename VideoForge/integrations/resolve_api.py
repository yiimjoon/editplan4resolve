import os
import subprocess
import math
import logging
import random
import tempfile
import threading
from pathlib import Path
from typing import Any, List, Optional, Iterable, Dict

from VideoForge.agent.planner import EditPlan
from VideoForge.config.config_manager import Config

from VideoForge.integrations.resolve_python import _get_resolve

logger = logging.getLogger(__name__)
_LOGGED_MARKER_MISSING = False
_LOGGED_MARKER_FAILED = False


class ResolveAPI:
    """Convenience wrapper for DaVinciResolveScript API."""

    def __init__(self) -> None:
        """Initialize Resolve scripting instance."""
        self._main_thread_id = threading.get_ident()
        self._strict_thread_guard = os.environ.get("VIDEOFORGE_STRICT_RESOLVE_THREAD", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        self._non_main_threads_warned: set[int] = set()

        try:
            self._resolve = _get_resolve()
        except Exception as exc:
            raise RuntimeError(
                "DaVinci Resolve scripting API is not available. Start Resolve and enable scripting."
            ) from exc

    def _ensure_main_thread(self, operation: str) -> None:
        """Warn (or error) if Resolve API is called off the main thread."""
        current_thread_id = threading.get_ident()
        if current_thread_id == self._main_thread_id:
            return

        if current_thread_id not in self._non_main_threads_warned:
            logger.warning(
                "Resolve API called off main thread (op=%s, thread=%s, main=%s).",
                operation,
                current_thread_id,
                self._main_thread_id,
            )
            self._non_main_threads_warned.add(current_thread_id)

        if self._strict_thread_guard:
            raise RuntimeError(
                "Resolve API must be called from the main thread. "
                f"Operation={operation} thread={current_thread_id} main={self._main_thread_id}"
            )

    def is_available(self) -> bool:
        """Return True if Resolve scripting API is still reachable."""
        self._ensure_main_thread("is_available")
        try:
            manager = self._resolve.GetProjectManager()
            return manager is not None
        except Exception:
            return False

    def get_current_project(self) -> Any:
        """Return the current Resolve project."""
        self._ensure_main_thread("get_current_project")
        manager = self._resolve.GetProjectManager()
        project = manager.GetCurrentProject()
        if project is None:
            project = manager.CreateProject("VideoForge")
        return project

    def get_project_key(self) -> str:
        """Return a stable key for the current project/timeline."""
        self._ensure_main_thread("get_project_key")
        project = self.get_current_project()
        project_name = None
        for getter in ("GetName", "GetProjectName"):
            func = getattr(project, getter, None)
            if callable(func):
                try:
                    project_name = func()
                    break
                except Exception:
                    continue
        timeline_name = None
        try:
            timeline = self.get_current_timeline()
            for getter in ("GetName",):
                func = getattr(timeline, getter, None)
                if callable(func):
                    timeline_name = func()
                    break
        except Exception:
            timeline = None
        project_name = project_name or "ResolveProject"
        timeline_name = timeline_name or "Timeline"
        return f"{project_name}::{timeline_name}"

    def get_current_timeline(self) -> Any:
        """Return the active timeline."""
        self._ensure_main_thread("get_current_timeline")
        project = self.get_current_project()
        timeline = project.GetCurrentTimeline()
        if timeline is None:
            raise RuntimeError("No active timeline. Create or open a timeline first.")
        return timeline

    def get_selected_clips(self) -> List[Any]:
        """Return the currently selected timeline clips."""
        self._ensure_main_thread("get_selected_clips")
        timeline = self.get_current_timeline()
        selected: List[Any] = []
        track_count = int(timeline.GetTrackCount("video") or 0)
        for track_index in range(1, track_count + 1):
            items = timeline.GetItemListInTrack("video", track_index) or []
            for item in items:
                try:
                    if item.GetSelected():
                        selected.append(item)
                except Exception:
                    continue
        return selected

    def get_selected_items(self, track_types: Optional[List[str]] = None) -> List[Any]:
        """Return selected timeline items across track types."""
        self._ensure_main_thread("get_selected_items")
        timeline = self.get_current_timeline()
        getter = getattr(timeline, "GetSelectedItems", None)
        logger.info("Selection debug: GetSelectedItems callable=%s", callable(getter))
        if callable(getter):
            try:
                raw = getter()
                selected = self._flatten_selected_items(raw)
                logger.info(
                    "Selection debug: GetSelectedItems() returned %d item(s) (type=%s)",
                    len(selected),
                    type(raw).__name__,
                )
                if selected:
                    return selected
            except Exception:
                logger.info("Selection debug: GetSelectedItems() failed", exc_info=True)
                pass
            if track_types:
                for track_type in track_types:
                    try:
                        raw = getter(track_type)
                        selected = self._flatten_selected_items(raw)
                        logger.info(
                            "Selection debug: GetSelectedItems(%s) returned %d item(s) (type=%s)",
                            track_type,
                            len(selected),
                            type(raw).__name__,
                        )
                        if selected:
                            return selected
                    except Exception:
                        logger.info(
                            "Selection debug: GetSelectedItems(%s) failed",
                            track_type,
                            exc_info=True,
                        )
                        continue
        selected: List[Any] = []
        track_types = track_types or ["video", "audio"]
        for track_type in track_types:
            track_count = int(timeline.GetTrackCount(track_type) or 0)
            logger.info(
                "Selection debug: scanning track_type=%s tracks=%d", track_type, track_count
            )
            for track_index in range(1, track_count + 1):
                items = timeline.GetItemListInTrack(track_type, track_index) or []
                for item in items:
                    try:
                        if item.GetSelected():
                            selected.append(item)
                    except Exception:
                        continue
        logger.info("Selection debug: fallback selected=%d item(s)", len(selected))
        return selected

    def get_items_at_playhead(self, track_types: Optional[List[str]] = None) -> List[Any]:
        """Return timeline items that overlap the playhead frame."""
        self._ensure_main_thread("get_items_at_playhead")
        timeline = self.get_current_timeline()
        frame = self._get_playhead_frame(timeline)
        if frame is None:
            return []
        items_at_frame: List[Any] = []
        track_types = track_types or ["video"]
        for track_type in track_types:
            track_count = int(timeline.GetTrackCount(track_type) or 0)
            for track_index in range(1, track_count + 1):
                items = timeline.GetItemListInTrack(track_type, track_index) or []
                for item in items:
                    start, end = self._get_item_range(item)
                    if start is None or end is None:
                        continue
                    if start <= frame <= end:
                        items_at_frame.append(item)
        return items_at_frame

    @staticmethod
    def _flatten_selected_items(raw: Any) -> List[Any]:
        items: List[Any] = []
        if raw is None:
            return items
        if isinstance(raw, dict):
            for value in raw.values():
                items.extend(ResolveAPI._flatten_selected_items(value))
            return items
        if isinstance(raw, list):
            for value in raw:
                items.extend(ResolveAPI._flatten_selected_items(value))
            return items
        items.append(raw)
        return items

    def get_item_start_frame(self, item: Any) -> Optional[int]:
        """Return timeline start frame for a clip."""
        self._ensure_main_thread("get_item_start_frame")
        start, _end = self._get_item_range(item)
        return int(start) if start is not None else None

    def get_selected_clip_paths(self) -> List[str]:
        """Return file paths for selected timeline clips."""
        paths: List[str] = []
        for item in self.get_selected_items():
            path = self.get_item_media_path(item)
            if path:
                paths.append(path)
        return paths

    def get_primary_clip(self) -> Optional[Any]:
        """Return selected clip or fallback to the clip under the playhead."""
        self._ensure_main_thread("get_primary_clip")
        selected = self.get_selected_clips()
        if selected:
            return selected[0]
        return self.get_clip_at_playhead()

    def create_subtitles_from_audio(self, language: str = "auto") -> bool:
        """Run Resolve Studio auto-captioning on the current timeline."""
        self._ensure_main_thread("create_subtitles_from_audio")
        timeline = self.get_current_timeline()
        resolve = self._resolve
        settings = {}
        language_key = getattr(resolve, "SUBTITLE_LANGUAGE", None)
        if language_key is not None:
            lang_map = {
                "auto": getattr(resolve, "AUTO_CAPTION_AUTO", None),
                "en": getattr(resolve, "AUTO_CAPTION_ENGLISH", None),
                "ko": getattr(resolve, "AUTO_CAPTION_KOREAN", None),
            }
            lang_value = lang_map.get(language)
            if lang_value is not None:
                settings[language_key] = lang_value

        preset_key = getattr(resolve, "SUBTITLE_CAPTION_PRESET", None)
        default_preset = getattr(resolve, "AUTO_CAPTION_SUBTITLE_DEFAULT", None)
        if preset_key is not None and default_preset is not None:
            settings[preset_key] = default_preset

        if settings:
            return bool(timeline.CreateSubtitlesFromAudio(settings))
        return bool(timeline.CreateSubtitlesFromAudio())

    def get_subtitle_items(self) -> List[Any]:
        """Return subtitle timeline items across all subtitle tracks."""
        self._ensure_main_thread("get_subtitle_items")
        timeline = self.get_current_timeline()
        items: List[Any] = []
        track_count = int(timeline.GetTrackCount("subtitle") or 0)
        for track_index in range(1, track_count + 1):
            items.extend(timeline.GetItemListInTrack("subtitle", track_index) or [])
        return items

    def export_subtitles_as_sentences(
        self,
        clip_range: tuple[float, float] | None = None,
    ) -> List[dict]:
        """Extract subtitle items into sentence-like dicts (t0/t1/text)."""
        self._ensure_main_thread("export_subtitles_as_sentences")
        fps = self.get_timeline_fps()
        sentences: List[dict] = []
        min_t = clip_range[0] if clip_range else None
        max_t = clip_range[1] if clip_range else None
        for item in self.get_subtitle_items():
            t0, t1 = self._item_time_range_seconds(item, fps)
            if min_t is not None and max_t is not None:
                if t1 < min_t or t0 > max_t:
                    continue
            text = self._subtitle_text(item)
            if text is None:
                continue
            sentences.append(
                {
                    "t0": t0,
                    "t1": t1,
                    "text": text,
                    "confidence": None,
                    "segment_id": 0,
                    "metadata": {"source": "resolve_ai"},
                }
            )
        return sentences

    def get_item_range_seconds(self, item: Any) -> tuple[float, float] | None:
        """Return timeline item range as seconds (t0, t1)."""
        self._ensure_main_thread("get_item_range_seconds")
        fps = self.get_timeline_fps()
        start, end = self._get_item_range(item)
        if start is None or end is None or fps <= 0:
            return None
        return float(start) / float(fps), float(end) / float(fps)

    def get_clip_source_range_seconds(self, item: Any) -> tuple[float, float] | None:
        """Return source in/out range for a timeline item in seconds."""
        self._ensure_main_thread("get_clip_source_range_seconds")
        fps = self.get_timeline_fps()
        if fps <= 0:
            return None

        left_offset = self._safe_call(item, "GetLeftOffset")
        duration = self._safe_call(item, "GetDuration")
        if left_offset is not None and duration is not None:
            try:
                start = float(left_offset) / float(fps)
                end = (float(left_offset) + float(duration)) / float(fps)
                return start, end
            except Exception:
                pass

        source_start = self._safe_call(item, "GetSourceStartFrame")
        source_end = self._safe_call(item, "GetSourceEndFrame")
        if source_start is not None and source_end is not None:
            try:
                return float(source_start) / float(fps), float(source_end) / float(fps)
            except Exception:
                pass

        return self.get_item_range_seconds(item)

    @staticmethod
    def _safe_call(item: Any, method_name: str):
        method = getattr(item, method_name, None)
        if callable(method):
            try:
                return method()
            except Exception:
                return None
        return None

    def debug_apply_silence_cuts(
        self,
        clip: Any,
        keep_segments: Iterable[Dict[str, float]],
        dry_run: bool = True,
    ) -> bool:
        """Debug whether we can split a timeline clip based on keep_segments."""
        timeline = self.get_current_timeline()
        fps = self.get_timeline_fps()
        if fps <= 0:
            logger.info("Silence cut debug: invalid FPS")
            return False

        timeline_methods = [m for m in dir(timeline) if "split" in m.lower() or "cut" in m.lower()]
        clip_methods = [m for m in dir(clip) if "split" in m.lower() or "cut" in m.lower()]
        logger.info("Silence cut debug: timeline split/cut methods: %s", timeline_methods)
        logger.info("Silence cut debug: clip split/cut methods: %s", clip_methods)

        source_range = self.get_clip_source_range_seconds(clip)
        timeline_range = self.get_item_range_seconds(clip)
        logger.info("Silence cut debug: clip timeline range: %s", timeline_range)
        logger.info("Silence cut debug: clip source range: %s", source_range)

        cuts_sec: List[float] = []
        for seg in keep_segments:
            cuts_sec.append(float(seg["t0"]))
            cuts_sec.append(float(seg["t1"]))
        cuts_sec = sorted({c for c in cuts_sec if c >= 0.0})
        logger.info("Silence cut debug: %d cut points (sec)", len(cuts_sec))
        logger.info("Silence cut debug: first cuts: %s", cuts_sec[:10])

        if dry_run:
            return True

        split_methods = [
            ("SplitClip", timeline),
            ("Split", clip),
        ]
        for name, obj in split_methods:
            method = getattr(obj, name, None)
            if callable(method):
                logger.info("Silence cut debug: attempting %s on %s", name, type(obj).__name__)
                try:
                    for cut in cuts_sec:
                        frame = int(round(cut * fps))
                        try:
                            if obj is timeline:
                                method(clip, frame)
                            else:
                                method(frame)
                        except Exception as exc:
                            logger.debug("Split failed at %s (%s): %s", cut, frame, exc)
                    return True
                except Exception as exc:
                    logger.info("Silence cut debug: %s failed: %s", name, exc)
        logger.info("Silence cut debug: no supported split method found")
        return False

    def apply_multicam_cuts(self, plan: EditPlan) -> Dict[str, Any]:
        """Apply multicam cut_and_switch_angle actions or export an EDL fallback."""
        self._ensure_main_thread("apply_multicam_cuts")
        timeline = self.get_current_timeline()
        fps = self.get_timeline_fps()

        actions = [a for a in plan.actions if a.action_type == "cut_and_switch_angle"]
        if not actions:
            return {"success": False, "error": "No multicam actions in plan."}

        track_count = int(timeline.GetTrackCount("video") or 0)
        if track_count <= 0:
            return {"success": False, "error": "No video tracks available."}

        split_available = callable(getattr(timeline, "SplitClip", None))
        if not split_available:
            split_available = self._resolve_split_method(timeline, track_count) is not None
        disable_method = self._resolve_disable_method(timeline, track_count)
        if not split_available or disable_method is None:
            logger.info(
                "Multicam cut: split_available=%s disable_method=%s; attempting new timeline build.",
                split_available,
                disable_method,
            )
            timeline_result = self._build_multicam_program_timeline(plan, fps)
            if timeline_result.get("success"):
                logger.info(
                    "Multicam new timeline created: %s",
                    timeline_result.get("timeline_name"),
                )
                return timeline_result
            logger.warning("Multicam new timeline failed: %s", timeline_result)
            return self._export_multicam_edl(plan, fps)

        actions_sorted = sorted(
            actions,
            key=lambda action: float(action.params.get("start_sec", 0.0)),
        )

        for action in actions_sorted:
            params = action.params or {}
            start_sec = float(params.get("start_sec", 0.0))
            end_sec = float(params.get("end_sec", start_sec))
            start_frame = int(params.get("cut_frame", int(start_sec * fps)))
            end_frame = int(end_sec * fps)
            if end_frame <= start_frame:
                end_frame = start_frame + 1
            target_track = int(params.get("target_angle_track", 0)) + 1

            self._split_all_tracks(timeline, track_count, start_frame)
            self._split_all_tracks(timeline, track_count, end_frame)

            if disable_method is None:
                return self._export_multicam_edl(plan, fps)

            for track_index in range(1, track_count + 1):
                if track_index == target_track:
                    continue
                items = timeline.GetItemListInTrack("video", track_index) or []
                for item in items:
                    s, e = self._get_item_range(item)
                    if s is None or e is None:
                        continue
                    if e <= start_frame or s >= end_frame:
                        continue
                    if not self._set_clip_enabled(item, disable_method, False):
                        return self._export_multicam_edl(plan, fps)

        return {"success": True}

    def _build_multicam_timeline(self, plan: EditPlan, fps: float) -> Dict[str, Any]:
        project = self.get_current_project()
        current_timeline = self.get_current_timeline()
        current_name = "Timeline"
        getter = getattr(current_timeline, "GetName", None)
        if callable(getter):
            try:
                current_name = str(getter())
            except Exception:
                pass
        base_name = f"{current_name} (Multicam)"
        timeline_name = self._unique_timeline_name(project, base_name)

        media_pool = self.get_media_pool()
        source_items = self._snapshot_video_track_items(current_timeline)
        program_track_index = self._resolve_program_track_index(source_items)
        create_method = getattr(media_pool, "CreateEmptyTimeline", None)
        create_context = "media_pool.CreateEmptyTimeline"
        if not callable(create_method):
            create_method = getattr(project, "CreateEmptyTimeline", None)
            create_context = "project.CreateEmptyTimeline"
        if not callable(create_method):
            create_method = getattr(project, "CreateTimeline", None)
            create_context = "project.CreateTimeline"
        if not callable(create_method):
            logger.warning(
                "Multicam timeline creation not available (CreateEmptyTimeline/CreateTimeline missing)."
            )
            return {"success": False, "error": "CreateEmptyTimeline not available"}

        try:
            new_timeline = create_method(timeline_name)
        except Exception as exc:
            logger.warning("Multicam timeline creation failed (%s): %s", create_context, exc)
            return {"success": False, "error": f"Failed to create timeline: {exc}"}
        if not new_timeline:
            logger.warning("Multicam timeline creation returned None for %s", timeline_name)
            return {"success": False, "error": "Timeline creation returned None"}

        set_current = getattr(project, "SetCurrentTimeline", None)
        if callable(set_current):
            try:
                set_current(new_timeline)
            except Exception:
                pass
        logger.info("Multicam timeline created: %s", timeline_name)
        try:
            current_name = ""
            current_tl = self.get_current_timeline()
            name_getter = getattr(current_tl, "GetName", None)
            if callable(name_getter):
                current_name = str(name_getter())
            logger.info("Multicam timeline current: %s", current_name or "<unknown>")
        except Exception as exc:
            logger.info("Multicam timeline current read failed: %s", exc)

        timeline_start = self.get_timeline_start_frame()
        logger.info("Multicam timeline: start_frame=%s", timeline_start)
        audio_mode, audio_track = self._get_multicam_audio_settings()
        if audio_mode == "fixed_track":
            logger.info(
                "Multicam timeline: using fixed audio from V%s",
                audio_track,
            )
        base_offset_sec = 0.0
        if source_items:
            base_offset_sec = min(
                float(item.get("timeline_start", 0.0)) for item in source_items
            )
            logger.info(
                "Multicam timeline: rebuilding %s source items (base_offset=%.2fs)",
                len(source_items),
                base_offset_sec,
            )
        for item in source_items:
            media_path = item.get("media_path")
            if not media_path:
                logger.warning(
                    "Multicam timeline: missing media path for track %s",
                    item.get("track_index"),
                )
                continue
            media_item = self.import_media_if_needed(str(media_path))
            if media_item is None:
                logger.warning("Multicam timeline: import failed for %s", media_path)
                continue
            clip_fps = self._get_media_item_fps(media_item, fps)
            if abs(clip_fps - fps) > 0.01:
                logger.info(
                    "Multicam timeline: clip fps %.3f differs from timeline fps %.3f (%s)",
                    clip_fps,
                    fps,
                    media_path,
                )
            relative_start = float(item.get("timeline_start", 0.0)) - base_offset_sec
            record_frame = int(round(timeline_start + relative_start * fps))
            source_in_frame = int(round(float(item["source_start"]) * clip_fps))
            source_out_frame = int(round(float(item["source_end"]) * clip_fps))
            inserted = self.insert_clip_at_position_with_range(
                "video",
                int(item["track_index"]),
                media_item,
                record_frame,
                source_in_frame,
                source_out_frame,
            )
            if inserted is None:
                logger.warning(
                    "Multicam timeline: base insert failed for %s on track %s",
                    media_path,
                    item.get("track_index"),
                )
        actions = [a for a in plan.actions if a.action_type == "cut_and_switch_angle"]
        actions.sort(key=lambda action: float(action.params.get("start_sec", 0.0)))

        errors: List[str] = []
        cursor_frame = int(timeline_start)
        def _safe_float(value: Any, fallback: float) -> float:
            if value is None:
                return float(fallback)
            try:
                return float(value)
            except Exception:
                return float(fallback)

        for action in actions:
            params = action.params or {}
            start_sec = _safe_float(params.get("start_sec", 0.0), 0.0)
            end_sec = _safe_float(params.get("end_sec", start_sec), start_sec)
            if end_sec <= start_sec:
                end_sec = start_sec + 0.04
            source_path = self._resolve_multicam_source_path(params)
            if not source_path:
                message = f"Missing source path for segment {start_sec:.2f}-{end_sec:.2f}"
                logger.warning("Multicam timeline: %s", message)
                errors.append(message)
                continue
            logger.info(
                "Multicam timeline: segment %.2f-%.2f source=%s",
                start_sec,
                end_sec,
                source_path,
            )
            media_item = self.import_media_if_needed(str(source_path))
            if media_item is None:
                message = f"Import failed for {source_path}"
                logger.warning("Multicam timeline: %s", message)
                errors.append(message)
                continue

            clip_fps = self._get_media_item_fps(media_item, fps)
            if abs(clip_fps - fps) > 0.01:
                logger.info(
                    "Multicam timeline: clip fps %.3f differs from timeline fps %.3f (%s)",
                    clip_fps,
                    fps,
                    source_path,
                )
            source_in_sec = _safe_float(params.get("source_in_sec", start_sec), start_sec)
            source_out_sec = _safe_float(params.get("source_out_sec", end_sec), end_sec)
            if source_out_sec <= source_in_sec:
                source_out_sec = source_in_sec + 0.04

            audio_source_path = source_path
            audio_source_in_sec = source_in_sec
            audio_source_out_sec = source_out_sec
            audio_media_item = media_item
            audio_clip_fps = clip_fps
            if audio_mode == "fixed_track":
                audio_params = self._resolve_multicam_source_range(params, audio_track)
                if audio_params:
                    audio_source_path = audio_params.get("source_path") or audio_source_path
                    audio_source_in_sec = _safe_float(
                        audio_params.get("source_in_sec", audio_source_in_sec),
                        audio_source_in_sec,
                    )
                    audio_source_out_sec = _safe_float(
                        audio_params.get("source_out_sec", audio_source_out_sec),
                        audio_source_out_sec,
                    )
                else:
                    logger.warning(
                        "Multicam timeline: missing audio source for V%s at %.2fs",
                        audio_track,
                        start_sec,
                    )
            if audio_source_out_sec <= audio_source_in_sec:
                audio_source_out_sec = audio_source_in_sec + 0.04
            if audio_source_path != source_path:
                audio_media_item = self.import_media_if_needed(str(audio_source_path))
                if audio_media_item is None:
                    logger.warning(
                        "Multicam timeline: audio import failed for %s; using video audio",
                        audio_source_path,
                    )
                    audio_media_item = media_item
                    audio_source_path = source_path
                    audio_source_in_sec = source_in_sec
                    audio_source_out_sec = source_out_sec
            if audio_media_item is not None:
                audio_clip_fps = self._get_media_item_fps(audio_media_item, fps)

            duration_frames = max(1, int(round((end_sec - start_sec) * fps)))
            desired_frame = int(round(timeline_start + start_sec * fps))
            if cursor_frame == int(timeline_start):
                cursor_frame = desired_frame
            record_frame = cursor_frame
            if record_frame != desired_frame:
                logger.info(
                    "Multicam timeline: drifted record_frame from %s to %s",
                    desired_frame,
                    record_frame,
                )
            source_in_frame = int(round(source_in_sec * clip_fps))
            source_duration_frames = max(
                1, int(round((duration_frames / fps) * clip_fps))
            )
            source_out_frame = source_in_frame + source_duration_frames
            timeline_duration_frames = max(
                1, int(math.floor(source_duration_frames * fps / clip_fps))
            )
            if timeline_duration_frames != duration_frames:
                logger.info(
                    "Multicam timeline: duration %s frames adjusted to %s for clip fps %.3f",
                    duration_frames,
                    timeline_duration_frames,
                    clip_fps,
                )
            cursor_frame = record_frame + timeline_duration_frames
            logger.info(
                "Multicam timeline: record_frame=%s source_frames=%s-%s",
                record_frame,
                source_in_frame,
                source_out_frame,
            )

            inserted_video = self.insert_clip_at_position_with_range(
                "video",
                program_track_index,
                media_item,
                record_frame,
                source_in_frame,
                source_out_frame,
            )
            if inserted_video is None:
                message = f"Insert failed for {source_path} at {start_sec:.2f}s"
                logger.warning("Multicam timeline: %s", message)
                errors.append(message)
                continue
            logger.info(
                "Multicam timeline: inserted video at %s (frames %s-%s)",
                record_frame,
                source_in_frame,
                source_out_frame,
            )

            audio_source_in_frame = int(round(audio_source_in_sec * audio_clip_fps))
            audio_source_duration_frames = max(
                1, int(round((timeline_duration_frames / fps) * audio_clip_fps))
            )
            audio_source_out_frame = audio_source_in_frame + audio_source_duration_frames
            if audio_source_out_sec > audio_source_in_sec:
                max_out_frame = int(round(audio_source_out_sec * audio_clip_fps))
                if audio_source_out_frame > max_out_frame:
                    audio_source_out_frame = max_out_frame
                    if audio_source_out_frame <= audio_source_in_frame:
                        audio_source_out_frame = audio_source_in_frame + 1
            if audio_media_item is not None:
                self.insert_clip_at_position_with_range(
                    "audio",
                    1,
                    audio_media_item,
                    record_frame,
                    audio_source_in_frame,
                    audio_source_out_frame,
                )

        try:
            track_counts = []
            current_tl = self.get_current_timeline()
            total_tracks = int(current_tl.GetTrackCount("video") or 0)
            for track_index in range(1, total_tracks + 1):
                items = current_tl.GetItemListInTrack("video", track_index) or []
                track_counts.append(f"V{track_index}={len(items)}")
            logger.info("Multicam timeline track counts: %s", ", ".join(track_counts))
        except Exception as exc:
            logger.info("Multicam timeline track count read failed: %s", exc)

        try:
            current_tl = self.get_current_timeline()
            program_items = current_tl.GetItemListInTrack("video", program_track_index) or []
        except Exception:
            program_items = []

        if not program_items:
            logger.warning(
                "Multicam timeline: program track empty; building program-only timeline."
            )
            return self._build_multicam_program_timeline(plan, fps)

        if errors:
            return {
                "success": False,
                "status": "new_timeline_failed",
                "timeline_name": timeline_name,
                "error": "; ".join(errors[:3]),
                "details": errors,
            }

        return {"success": True, "status": "new_timeline", "timeline_name": timeline_name}

    def _snapshot_video_track_items(self, timeline: Any) -> List[Dict[str, Any]]:
        track_count = int(timeline.GetTrackCount("video") or 0)
        items: List[Dict[str, Any]] = []
        for track_index in range(1, track_count + 1):
            for item in timeline.GetItemListInTrack("video", track_index) or []:
                timeline_range = self.get_item_range_seconds(item)
                if not timeline_range:
                    continue
                source_range = self.get_clip_source_range_seconds(item) or timeline_range
                media_path = self.get_item_media_path(item)
                items.append(
                    {
                        "track_index": int(track_index),
                        "timeline_start": float(timeline_range[0]),
                        "timeline_end": float(timeline_range[1]),
                        "source_start": float(source_range[0]),
                        "source_end": float(source_range[1]),
                        "media_path": media_path,
                    }
                )
        return items

    def _build_multicam_program_timeline(self, plan: EditPlan, fps: float) -> Dict[str, Any]:
        project = self.get_current_project()
        current_timeline = self.get_current_timeline()
        current_name = "Timeline"
        getter = getattr(current_timeline, "GetName", None)
        if callable(getter):
            try:
                current_name = str(getter())
            except Exception:
                pass
        base_name = f"{current_name} (Multicam Program)"
        timeline_name = self._unique_timeline_name(project, base_name)

        media_pool = self.get_media_pool()
        create_method = getattr(media_pool, "CreateEmptyTimeline", None)
        create_context = "media_pool.CreateEmptyTimeline"
        if not callable(create_method):
            create_method = getattr(project, "CreateEmptyTimeline", None)
            create_context = "project.CreateEmptyTimeline"
        if not callable(create_method):
            create_method = getattr(project, "CreateTimeline", None)
            create_context = "project.CreateTimeline"
        if not callable(create_method):
            logger.warning(
                "Program timeline creation not available (CreateEmptyTimeline/CreateTimeline missing)."
            )
            return {"success": False, "error": "CreateEmptyTimeline not available"}

        try:
            new_timeline = create_method(timeline_name)
        except Exception as exc:
            logger.warning("Program timeline creation failed (%s): %s", create_context, exc)
            return {"success": False, "error": f"Failed to create timeline: {exc}"}
        if not new_timeline:
            logger.warning("Program timeline creation returned None for %s", timeline_name)
            return {"success": False, "error": "Timeline creation returned None"}

        set_current = getattr(project, "SetCurrentTimeline", None)
        if callable(set_current):
            try:
                set_current(new_timeline)
            except Exception:
                pass
        logger.info("Multicam program timeline created: %s", timeline_name)

        timeline_start = self.get_timeline_start_frame()
        audio_mode, audio_track = self._get_multicam_audio_settings()
        if audio_mode == "fixed_track":
            logger.info(
                "Multicam program timeline: using fixed audio from V%s",
                audio_track,
            )
        actions = [a for a in plan.actions if a.action_type == "cut_and_switch_angle"]
        actions.sort(key=lambda action: float(action.params.get("start_sec", 0.0)))

        errors: List[str] = []
        cursor_frame = int(timeline_start)

        def _safe_float(value: Any, fallback: float) -> float:
            if value is None:
                return float(fallback)
            try:
                return float(value)
            except Exception:
                return float(fallback)

        for action in actions:
            params = action.params or {}
            start_sec = _safe_float(params.get("start_sec", 0.0), 0.0)
            end_sec = _safe_float(params.get("end_sec", start_sec), start_sec)
            if end_sec <= start_sec:
                end_sec = start_sec + 0.04
            source_path = self._resolve_multicam_source_path(params)
            if not source_path:
                message = f"Missing source path for segment {start_sec:.2f}-{end_sec:.2f}"
                logger.warning("Multicam program timeline: %s", message)
                errors.append(message)
                continue

            media_item = self.import_media_if_needed(str(source_path))
            if media_item is None:
                message = f"Import failed for {source_path}"
                logger.warning("Multicam program timeline: %s", message)
                errors.append(message)
                continue

            clip_fps = self._get_media_item_fps(media_item, fps)
            if abs(clip_fps - fps) > 0.01:
                logger.info(
                    "Multicam program timeline: clip fps %.3f differs from timeline fps %.3f (%s)",
                    clip_fps,
                    fps,
                    source_path,
                )
            source_in_sec = _safe_float(params.get("source_in_sec", start_sec), start_sec)
            source_out_sec = _safe_float(params.get("source_out_sec", end_sec), end_sec)
            if source_out_sec <= source_in_sec:
                source_out_sec = source_in_sec + 0.04

            audio_source_path = source_path
            audio_source_in_sec = source_in_sec
            audio_source_out_sec = source_out_sec
            audio_media_item = media_item
            audio_clip_fps = clip_fps
            if audio_mode == "fixed_track":
                audio_params = self._resolve_multicam_source_range(params, audio_track)
                if audio_params:
                    audio_source_path = audio_params.get("source_path") or audio_source_path
                    audio_source_in_sec = _safe_float(
                        audio_params.get("source_in_sec", audio_source_in_sec),
                        audio_source_in_sec,
                    )
                    audio_source_out_sec = _safe_float(
                        audio_params.get("source_out_sec", audio_source_out_sec),
                        audio_source_out_sec,
                    )
                else:
                    logger.warning(
                        "Multicam program timeline: missing audio source for V%s at %.2fs",
                        audio_track,
                        start_sec,
                    )
            if audio_source_out_sec <= audio_source_in_sec:
                audio_source_out_sec = audio_source_in_sec + 0.04
            if audio_source_path != source_path:
                audio_media_item = self.import_media_if_needed(str(audio_source_path))
                if audio_media_item is None:
                    logger.warning(
                        "Multicam program timeline: audio import failed for %s; using video audio",
                        audio_source_path,
                    )
                    audio_media_item = media_item
                    audio_source_path = source_path
                    audio_source_in_sec = source_in_sec
                    audio_source_out_sec = source_out_sec
            if audio_media_item is not None:
                audio_clip_fps = self._get_media_item_fps(audio_media_item, fps)

            duration_frames = max(1, int(round((end_sec - start_sec) * fps)))
            desired_frame = int(round(timeline_start + start_sec * fps))
            if cursor_frame == int(timeline_start):
                cursor_frame = desired_frame
            record_frame = cursor_frame
            if record_frame != desired_frame:
                logger.info(
                    "Multicam program timeline: drifted record_frame from %s to %s",
                    desired_frame,
                    record_frame,
                )
            source_in_frame = int(round(source_in_sec * clip_fps))
            source_duration_frames = max(
                1, int(round((duration_frames / fps) * clip_fps))
            )
            source_out_frame = source_in_frame + source_duration_frames
            timeline_duration_frames = max(
                1, int(math.floor(source_duration_frames * fps / clip_fps))
            )
            if timeline_duration_frames != duration_frames:
                logger.info(
                    "Multicam program timeline: duration %s frames adjusted to %s for clip fps %.3f",
                    duration_frames,
                    timeline_duration_frames,
                    clip_fps,
                )
            cursor_frame = record_frame + timeline_duration_frames

            inserted_video = self.insert_clip_at_position_with_range(
                "video",
                1,
                media_item,
                record_frame,
                source_in_frame,
                source_out_frame,
            )
            if inserted_video is None:
                message = f"Insert failed for {source_path} at {start_sec:.2f}s"
                logger.warning("Multicam program timeline: %s", message)
                errors.append(message)
                continue
            audio_source_in_frame = int(round(audio_source_in_sec * audio_clip_fps))
            audio_source_duration_frames = max(
                1, int(round((timeline_duration_frames / fps) * audio_clip_fps))
            )
            audio_source_out_frame = audio_source_in_frame + audio_source_duration_frames
            if audio_source_out_sec > audio_source_in_sec:
                max_out_frame = int(round(audio_source_out_sec * audio_clip_fps))
                if audio_source_out_frame > max_out_frame:
                    audio_source_out_frame = max_out_frame
                    if audio_source_out_frame <= audio_source_in_frame:
                        audio_source_out_frame = audio_source_in_frame + 1
            if audio_media_item is not None:
                self.insert_clip_at_position_with_range(
                    "audio",
                    1,
                    audio_media_item,
                    record_frame,
                    audio_source_in_frame,
                    audio_source_out_frame,
                )

        if errors:
            return {
                "success": False,
                "status": "program_timeline_failed",
                "timeline_name": timeline_name,
                "error": "; ".join(errors[:3]),
                "details": errors,
            }

        return {"success": True, "status": "program_timeline", "timeline_name": timeline_name}

    @staticmethod
    def _resolve_program_track_index(items: List[Dict[str, Any]]) -> int:
        if not items:
            return 1
        max_track = max(int(item.get("track_index", 1)) for item in items)
        return max_track + 1

    @staticmethod
    def _resolve_multicam_source_path(params: Dict[str, Any]) -> Optional[str]:
        if not isinstance(params, dict):
            return None
        direct_path = params.get("source_path")
        if isinstance(direct_path, str) and direct_path:
            return direct_path
        source_paths = params.get("source_paths")
        try:
            target = int(params.get("target_angle_track", 0))
        except Exception:
            target = 0
        if isinstance(source_paths, list) and 0 <= target < len(source_paths):
            candidate = source_paths[target]
            if isinstance(candidate, str) and candidate:
                return candidate
        return None

    @staticmethod
    def _resolve_multicam_source_range(
        params: Dict[str, Any], track_index: int
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(params, dict):
            return None
        ranges = params.get("source_ranges")
        if not isinstance(ranges, list):
            return None
        for entry in ranges:
            if not isinstance(entry, dict):
                continue
            try:
                if int(entry.get("track_index", 0)) != int(track_index):
                    continue
            except Exception:
                continue
            path = entry.get("source_path")
            if isinstance(path, str) and path:
                return entry
        return None

    @staticmethod
    def _get_multicam_audio_settings() -> tuple[str, int]:
        mode = str(Config.get("multicam_audio_mode", "per_cut")).strip().lower()
        if mode not in ("per_cut", "fixed_track"):
            mode = "per_cut"
        try:
            track_index = int(Config.get("multicam_audio_track", 1))
        except Exception:
            track_index = 1
        if track_index < 1:
            track_index = 1
        return mode, track_index

    @staticmethod
    def _unique_timeline_name(project: Any, base_name: str) -> str:
        names = set()
        count = 0
        getter = getattr(project, "GetTimelineCount", None)
        if callable(getter):
            try:
                count = int(getter() or 0)
            except Exception:
                count = 0
        if count:
            by_index = getattr(project, "GetTimelineByIndex", None)
            if callable(by_index):
                for idx in range(1, count + 1):
                    try:
                        timeline = by_index(idx)
                    except Exception:
                        timeline = None
                    if timeline is None:
                        continue
                    name_getter = getattr(timeline, "GetName", None)
                    if callable(name_getter):
                        try:
                            names.add(str(name_getter()))
                        except Exception:
                            continue
        if base_name not in names:
            return base_name
        suffix = 2
        while f"{base_name} {suffix}" in names:
            suffix += 1
        return f"{base_name} {suffix}"

    def _split_all_tracks(self, timeline: Any, track_count: int, frame: int) -> None:
        for track_index in range(1, track_count + 1):
            items = timeline.GetItemListInTrack("video", track_index) or []
            for item in items:
                s, e = self._get_item_range(item)
                if s is None or e is None:
                    continue
                if s < frame < e:
                    self._split_item_at_frame(timeline, item, frame)

    def _find_item_in_track_at_frame(
        self,
        timeline: Any,
        track_type: str,
        track_index: int,
        frame: int,
        media_path: Optional[str] = None,
    ) -> Optional[Any]:
        items = timeline.GetItemListInTrack(track_type, int(track_index)) or []
        for item in items:
            s, e = self._get_item_range(item)
            if s is None or e is None:
                continue
            if not (s <= frame <= e):
                continue
            if media_path:
                item_path = self._timeline_item_media_path(item)
                if not item_path:
                    continue
                if str(item_path).lower() != str(media_path).lower():
                    continue
            return item
        return None

    def _find_items_in_track_at_frame(
        self,
        timeline: Any,
        track_type: str,
        track_index: int,
        frame: int,
        media_path: Optional[str] = None,
    ) -> List[Any]:
        items = timeline.GetItemListInTrack(track_type, int(track_index)) or []
        matches: List[Any] = []
        for item in items:
            s, e = self._get_item_range(item)
            if s is None or e is None:
                continue
            if not (s <= frame <= e):
                continue
            if media_path:
                item_path = self._timeline_item_media_path(item)
                if not item_path:
                    continue
                if str(item_path).lower() != str(media_path).lower():
                    continue
            matches.append(item)
        return matches

    def split_clip_at_frames(
        self, clip: Any, frames: List[int], media_path: Optional[str] = None
    ) -> int:
        """Split a single clip at the specified timeline frames."""
        self._ensure_main_thread("split_clip_at_frames")
        if not frames:
            return 0
        timeline = self.get_current_timeline()
        cut_frames = sorted({int(frame) for frame in frames if frame is not None})
        track_type = self._safe_call(clip, "GetTrackType") or "video"
        track_index = self._safe_call(clip, "GetTrackIndex")
        media_path = media_path or self._timeline_item_media_path(clip)
        clip_start, clip_end = self._get_item_range(clip)
        applied = 0
        for frame in cut_frames:
            target = None
            if track_index:
                target = self._find_item_in_track_at_frame(
                    timeline,
                    str(track_type),
                    int(track_index),
                    int(frame),
                    media_path=media_path,
                )
            if target is None and media_path:
                track_count = int(timeline.GetTrackCount(str(track_type)) or 0)
                for idx in range(1, track_count + 1):
                    target = self._find_item_in_track_at_frame(
                        timeline,
                        str(track_type),
                        idx,
                        int(frame),
                        media_path=media_path,
                    )
                    if target is not None:
                        break
            if target is None:
                track_count = int(timeline.GetTrackCount(str(track_type)) or 0)
                candidates: List[Any] = []
                for idx in range(1, track_count + 1):
                    candidates.extend(
                        self._find_items_in_track_at_frame(
                            timeline,
                            str(track_type),
                            idx,
                            int(frame),
                        )
                    )
                if candidates:
                    if clip_start is not None and clip_end is not None:
                        for cand in candidates:
                            s, e = self._get_item_range(cand)
                            if s == clip_start and e == clip_end:
                                target = cand
                                break
                    if target is None and len(candidates) == 1:
                        target = candidates[0]
            if target is None:
                target = clip
            if self._split_item_at_frame(timeline, target, int(frame), self.get_timeline_fps()):
                applied += 1
            else:
                s, e = self._get_item_range(target)
                logger.info(
                    "Split clip failed at frame=%s range=%s-%s track=%s index=%s path=%s",
                    frame,
                    s,
                    e,
                    track_type,
                    track_index,
                    media_path or "<unknown>",
                )
        return applied

    def split_clip_by_inserting_ranges(
        self,
        clip: Any,
        frames: List[int],
        media_path: Optional[str] = None,
        keep_audio: bool = False,
    ) -> int:
        """Fallback: delete the original clip and insert trimmed ranges per cut frame."""
        self._ensure_main_thread("split_clip_by_inserting_ranges")
        if not frames:
            return 0
        timeline = self.get_current_timeline()
        fps = self.get_timeline_fps()
        if fps <= 0:
            return 0
        start, end = self._get_item_range(clip)
        if start is None or end is None:
            return 0

        track_type = self._safe_call(clip, "GetTrackType") or "video"
        track_index = self._safe_call(clip, "GetTrackIndex")
        media_path = media_path or self._timeline_item_media_path(clip)
        pool_item = self._safe_call(clip, "GetMediaPoolItem")
        if pool_item is None and media_path:
            pool_item = self.import_media_if_needed(str(media_path))
        if pool_item is None:
            logger.info("Split clip fallback failed: media pool item unavailable for %s", media_path or "<unknown>")
            return 0

        if not track_index:
            track_count = int(timeline.GetTrackCount(str(track_type)) or 0)
            for idx in range(1, track_count + 1):
                items = timeline.GetItemListInTrack(str(track_type), idx) or []
                for item in items:
                    s, e = self._get_item_range(item)
                    if s is None or e is None:
                        continue
                    if s != int(start) or e != int(end):
                        continue
                    item_path = self._timeline_item_media_path(item)
                    if media_path and item_path and str(item_path).lower() != str(media_path).lower():
                        continue
                    track_index = idx
                    break
                if track_index:
                    break
        track_index = int(track_index or 1)

        source_range = self.get_clip_source_range_seconds(clip)
        if not source_range:
            return 0
        source_start_sec, source_end_sec = source_range
        source_start_frame = int(round(source_start_sec * fps))
        source_end_frame = int(round(source_end_sec * fps))

        cut_frames = sorted({int(frame) for frame in frames if frame is not None})
        valid = [f for f in cut_frames if int(start) < f < int(end)]
        if not valid:
            return 0
        pre_snapshot = self._snapshot_track_items(str(track_type))

        insert_track_index = int(track_index)
        if keep_audio and str(track_type).lower() == "video":
            insert_track_index = int(track_index) + 1
            self.ensure_track(str(track_type), int(insert_track_index))
            try:
                clip.SetProperty("Video Enabled", False)
            except Exception:
                pass
        else:
            delete_method = getattr(timeline, "DeleteClips", None)
            if callable(delete_method):
                try:
                    delete_method([clip])
                except Exception as exc:
                    logger.info("Split clip fallback: DeleteClips failed: %s", exc)

        boundaries = [int(start)] + valid + [int(end)]
        inserted = 0
        audio_deleted = 0
        for seg_start, seg_end in zip(boundaries, boundaries[1:]):
            if seg_end - seg_start < 1:
                continue
            source_seg_start = source_start_frame + (seg_start - int(start))
            source_seg_end = source_start_frame + (seg_end - int(start))
            if source_seg_end <= source_seg_start:
                continue
            if source_seg_start < source_start_frame:
                source_seg_start = source_start_frame
            if source_seg_end > source_end_frame:
                source_seg_end = source_end_frame
            if source_seg_end <= source_seg_start:
                continue
            inserted_item = self.insert_clip_at_position_with_range(
                str(track_type),
                int(insert_track_index),
                pool_item,
                int(seg_start),
                int(source_seg_start),
                int(source_seg_end),
            )
            if inserted_item is not None:
                inserted += 1
                if str(track_type).lower() == "video" and not keep_audio:
                    try:
                        inserted_item.SetProperty("Audio Enabled", False)
                    except Exception:
                        pass
                    audio_deleted += self._remove_audio_items_for_media_path(
                        media_path, int(seg_start), int(seg_end)
                    )
        post_snapshot = self._snapshot_track_items(str(track_type))
        created = 0
        for idx, ids in post_snapshot.items():
            before = pre_snapshot.get(idx, set())
            created += len(ids - before)

        if created == 0 and inserted == 0:
            logger.info(
                "Split clip fallback: no timeline items created; keeping original clip (%s)",
                media_path or "<unknown>",
            )
            existing = False
            track_count = int(timeline.GetTrackCount(str(track_type)) or 0)
            for idx in range(1, track_count + 1):
                items = timeline.GetItemListInTrack(str(track_type), idx) or []
                for item in items:
                    s, e = self._get_item_range(item)
                    if s is None or e is None:
                        continue
                    if s != int(start) or e != int(end):
                        continue
                    item_path = self._timeline_item_media_path(item)
                    if media_path and item_path and str(item_path).lower() != str(media_path).lower():
                        continue
                    existing = True
                    break
                if existing:
                    break
            if not existing:
                self.insert_clip_at_position_with_range(
                    str(track_type),
                    int(track_index),
                    pool_item,
                    int(start),
                    int(source_start_frame),
                    int(source_end_frame),
                )
            return 0

        total = inserted if inserted > 0 else created
        logger.info(
            "Split clip fallback inserted %s segment(s) for %s (audio_deleted=%s keep_audio=%s)",
            total,
            media_path or "<unknown>",
            audio_deleted,
            keep_audio,
        )
        return total

    def _remove_audio_items_for_media_path(
        self, media_path: Optional[str], start_frame: int, end_frame: int
    ) -> int:
        timeline = self.get_current_timeline()
        delete_method = getattr(timeline, "DeleteClips", None)
        if not callable(delete_method):
            return 0
        track_count = int(timeline.GetTrackCount("audio") or 0)
        to_delete: List[Any] = []
        for track_index in range(1, track_count + 1):
            items = timeline.GetItemListInTrack("audio", track_index) or []
            for item in items:
                s, e = self._get_item_range(item)
                if s is None or e is None:
                    continue
                if e < start_frame or s > end_frame:
                    continue
                item_path = self._timeline_item_media_path(item)
                if media_path and item_path and str(item_path).lower() != str(media_path).lower():
                    continue
                to_delete.append(item)
        if not to_delete:
            return 0
        try:
            delete_method(to_delete)
            return len(to_delete)
        except Exception as exc:
            logger.info("Split clip fallback: audio delete failed: %s", exc)
            return 0

    def get_items_by_media_path_in_range(
        self,
        media_path: Optional[str],
        track_type: str,
        start_frame: int,
        end_frame: int,
    ) -> List[Any]:
        """Return timeline items for a media path overlapping a frame range."""
        timeline = self.get_current_timeline()
        if not timeline or not media_path:
            return []
        track_count = int(timeline.GetTrackCount(str(track_type)) or 0)
        items: List[Any] = []
        for track_index in range(1, track_count + 1):
            track_items = timeline.GetItemListInTrack(str(track_type), track_index) or []
            for item in track_items:
                s, e = self._get_item_range(item)
                if s is None or e is None:
                    continue
                if e < start_frame or s > end_frame:
                    continue
                item_path = self._timeline_item_media_path(item)
                if not item_path:
                    continue
                if str(item_path).lower() != str(media_path).lower():
                    continue
                items.append(item)
        return items

    @staticmethod
    def _split_item_at_frame(
        timeline: Any, item: Any, frame: int, fps: Optional[float] = None
    ) -> bool:
        start, end = ResolveAPI._get_item_range(item)
        if start is None or end is None:
            return False
        if frame <= int(start) or frame >= int(end):
            return False
        rel_frame = int(frame) - int(start)
        split_clip = getattr(timeline, "SplitClip", None)
        if callable(split_clip):
            result = None
            try:
                result = split_clip(item, int(frame))
            except Exception:
                result = None
            if isinstance(result, bool):
                if result:
                    return True
            elif result is not None:
                return True
            try:
                result = split_clip(item, int(rel_frame))
                if isinstance(result, bool):
                    if result:
                        return True
                elif result is not None:
                    return True
            except Exception:
                pass
            if fps:
                try:
                    timecode = ResolveAPI._frames_to_timecode(int(frame), float(fps))
                    result = split_clip(item, timecode)
                    if isinstance(result, bool):
                        return bool(result)
                    if result is not None:
                        return True
                except Exception:
                    pass
            set_frame = getattr(timeline, "SetCurrentFrame", None)
            set_timecode = getattr(timeline, "SetCurrentTimecode", None)
            get_frame = getattr(timeline, "GetCurrentFrame", None)
            get_timecode = getattr(timeline, "GetCurrentTimecode", None)
            prev_frame = None
            prev_timecode = None
            if callable(get_frame):
                try:
                    prev_frame = int(get_frame())
                except Exception:
                    prev_frame = None
            if prev_frame is None and callable(get_timecode):
                try:
                    prev_timecode = str(get_timecode())
                except Exception:
                    prev_timecode = None
            if callable(set_frame):
                try:
                    set_frame(int(frame))
                except Exception:
                    pass
            elif callable(set_timecode) and fps:
                try:
                    set_timecode(ResolveAPI._frames_to_timecode(int(frame), float(fps)))
                except Exception:
                    pass
            try:
                result = split_clip(item)
                if isinstance(result, bool):
                    if result:
                        return True
                elif result is not None:
                    return True
            except Exception:
                pass
            try:
                result = split_clip()
                if isinstance(result, bool):
                    if result:
                        return True
                elif result is not None:
                    return True
            except Exception:
                pass
            finally:
                if prev_frame is not None and callable(set_frame):
                    try:
                        set_frame(prev_frame)
                    except Exception:
                        pass
                elif prev_timecode is not None and callable(set_timecode):
                    try:
                        set_timecode(prev_timecode)
                    except Exception:
                        pass
        split_item = getattr(item, "Split", None)
        if callable(split_item):
            try:
                result = split_item(int(rel_frame))
                if isinstance(result, bool):
                    return bool(result)
                return True
            except Exception:
                try:
                    result = split_item(int(frame))
                    if isinstance(result, bool):
                        return bool(result)
                    return True
                except Exception:
                    return False
        return False

    def _resolve_disable_method(self, timeline: Any, track_count: int) -> Optional[str]:
        for track_index in range(1, track_count + 1):
            items = timeline.GetItemListInTrack("video", track_index) or []
            for item in items:
                for name in ("SetClipEnabled", "SetEnabled"):
                    if callable(getattr(item, name, None)):
                        return name
        return None

    def _resolve_split_method(self, timeline: Any, track_count: int) -> Optional[str]:
        if callable(getattr(timeline, "SplitClip", None)):
            return "SplitClip"
        for track_index in range(1, track_count + 1):
            items = timeline.GetItemListInTrack("video", track_index) or []
            for item in items:
                if callable(getattr(item, "Split", None)):
                    return "Split"
        return None

    @staticmethod
    def _set_clip_enabled(item: Any, method_name: str, enabled: bool) -> bool:
        method = getattr(item, method_name, None)
        if not callable(method):
            return False
        try:
            method(bool(enabled))
            return True
        except Exception:
            return False

    def _export_multicam_edl(self, plan: EditPlan, fps: float) -> Dict[str, Any]:
        edl_setting = str(Config.get("multicam_edl_output_dir", "") or "").strip()
        logger.info("Multicam EDL output setting: %s", edl_setting or "<default>")
        if edl_setting:
            candidate = Path(edl_setting).expanduser()
            if candidate.suffix.lower() == ".edl":
                output_path = candidate
                output_dir = output_path.parent
            else:
                output_dir = candidate
                output_path = output_dir / "videoforge_multicam.edl"
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                logger.warning(
                    "Invalid multicam EDL output dir (%s); falling back to temp: %s",
                    output_dir,
                    exc,
                )
                output_dir = Path(tempfile.gettempdir())
                output_path = output_dir / "videoforge_multicam.edl"
        else:
            output_dir = Path(tempfile.gettempdir())
            output_path = output_dir / "videoforge_multicam.edl"
        timeline_start = self.get_timeline_start_frame()
        actions = [a for a in plan.actions if a.action_type == "cut_and_switch_angle"]
        actions.sort(key=lambda action: float(action.params.get("start_sec", 0.0)))

        lines = ["TITLE: VideoForge Multicam", "FCM: NON-DROP FRAME", ""]
        event_num = 1
        for action in actions:
            params = action.params or {}
            start_sec = float(params.get("start_sec", 0.0))
            end_sec = float(params.get("end_sec", start_sec))
            if end_sec <= start_sec:
                continue
            target_angle = int(params.get("target_angle_track", 0))
            source_clips = params.get("source_clips")
            clip_name = None
            if isinstance(source_clips, list) and 0 <= target_angle < len(source_clips):
                clip_name = str(source_clips[target_angle])
            if not clip_name:
                clip_name = f"Angle {target_angle + 1}"
            clip_name = self._sanitize_edl_name(clip_name)
            reel = self._format_edl_reel(clip_name, fallback=f"AX{target_angle + 1:02d}")

            source_in = self._format_timecode_seconds(start_sec, fps)
            source_out = self._format_timecode_seconds(end_sec, fps)
            record_in = self._format_timecode_frames(timeline_start + start_sec * fps, fps)
            record_out = self._format_timecode_frames(timeline_start + end_sec * fps, fps)

            lines.append(
                f"{event_num:03d}  {reel}V     C        "
                f"{source_in} {source_out} {record_in} {record_out}"
            )
            lines.append(f"* FROM CLIP NAME: {clip_name}")
            event_num += 1

        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.warning(
            "Resolve API multicam cut unsupported; exported EDL: %s",
            output_path,
        )
        message = (
            "Resolve API does not support programmatic multicam switching.\n"
            f"EDL file generated: {output_path}\n\n"
            "To apply cuts:\n"
            "1. File -> Import -> Timeline -> EDL\n"
            "2. Select 'videoforge_multicam.edl'\n"
            "3. Choose 'Replace existing timeline' or 'New timeline'\n"
            "4. Verify cuts alignment"
        )
        return {"success": False, "status": "edl_fallback", "edl_path": str(output_path), "message": message}

    @staticmethod
    def _sanitize_edl_name(name: str) -> str:
        safe = []
        for ch in name:
            if 32 <= ord(ch) < 127:
                safe.append(ch)
            else:
                safe.append("_")
        return "".join(safe)

    @staticmethod
    def _format_timecode_seconds(seconds: float, fps: float) -> str:
        total_frames = int(round(seconds * fps))
        return ResolveAPI._frames_to_timecode(total_frames, fps)

    @staticmethod
    def _format_timecode_frames(frames: float, fps: float) -> str:
        total_frames = int(round(frames))
        return ResolveAPI._frames_to_timecode(total_frames, fps)

    @staticmethod
    def _format_edl_reel(name: str, fallback: str) -> str:
        raw = "".join(ch for ch in name if ch.isalnum() and ord(ch) < 128)
        if not raw:
            raw = fallback
        raw = raw[:8]
        return raw.ljust(8)

    def replace_timeline_item(
        self,
        clip: Any,
        new_path: str,
        ripple: bool = True,
    ) -> bool:
        """Replace a timeline clip with a new media file at the same position."""
        timeline = self.get_current_timeline()
        fps = self.get_timeline_fps()
        start_frame, end_frame = self._get_item_range(clip)
        track_index = self._safe_call(clip, "GetTrackIndex")
        track_type = self._safe_call(clip, "GetTrackType") or "video"
        if track_index is None:
            track_index = 1
        if start_frame is None:
            logger.info("Replace clip failed: missing start frame")
            return False
        media_item = self.import_media_if_needed(new_path)
        if media_item is None:
            logger.info("Replace clip failed: import failed for %s", new_path)
            return False

        logger.info(
            "Replace clip debug: track=%s index=%s start=%s end=%s ripple=%s",
            track_type,
            track_index,
            start_frame,
            end_frame,
            ripple,
        )

        pre_items = timeline.GetItemListInTrack(track_type, int(track_index)) or []
        pre_ids = {id(item) for item in pre_items}

        insert_item = None
        try:
            insert_item = self.insert_clip_at_position(
                track_type,
                int(track_index),
                media_item,
                int(start_frame),
            )
        except Exception as exc:
            logger.info("Replace clip debug: insert failed: %s", exc)

        post_items = timeline.GetItemListInTrack(track_type, int(track_index)) or []
        new_items = [item for item in post_items if id(item) not in pre_ids]
        if new_items:
            insert_item = new_items[0] if len(new_items) == 1 else None
            if insert_item is None:
                # Prefer a new item with matching media path if multiple were added.
                for item in new_items:
                    media_path = self._timeline_item_media_path(item)
                    if media_path and str(media_path).lower() == str(new_path).lower():
                        insert_item = item
                        break
        if insert_item is None:
            insert_item = self._find_latest_item_by_media_path(track_type, int(track_index), new_path)

        if insert_item is None:
            logger.info("Replace clip debug: insert did not create a new item; leaving original clip intact")
            return False

        insert_range = self._get_item_range(insert_item)
        if insert_range[0] is None or insert_range[1] is None:
            logger.info("Replace clip debug: inserted item has no range; removing inserted item and aborting")
            delete_method = getattr(timeline, "DeleteClips", None)
            if callable(delete_method):
                try:
                    delete_method([insert_item])
                except Exception as exc:
                    logger.info("Replace clip debug: cleanup delete failed: %s", exc)
            return False

        insert_start = int(insert_range[0])
        if abs(insert_start - int(start_frame)) > max(2, int(round(fps * 0.25))):
            logger.info(
                "Replace clip debug: inserted item landed at %s (expected %s); removing inserted item and aborting",
                insert_start,
                start_frame,
            )
            delete_method = getattr(timeline, "DeleteClips", None)
            if callable(delete_method):
                try:
                    delete_method([insert_item])
                except Exception as exc:
                    logger.info("Replace clip debug: cleanup delete failed: %s", exc)
            return False

        timeline_methods = [m for m in dir(timeline) if "delete" in m.lower() or "remove" in m.lower()]
        clip_methods = [m for m in dir(clip) if "delete" in m.lower() or "remove" in m.lower()]
        logger.info("Replace clip debug: timeline delete/remove methods: %s", timeline_methods)
        logger.info("Replace clip debug: clip delete/remove methods: %s", clip_methods)

        delete_candidates = [
            ("DeleteClips", timeline, [clip]),
            ("DeleteItems", timeline, [clip]),
            ("RemoveItems", timeline, [clip]),
            ("DeleteClip", timeline, clip),
            ("RemoveClip", timeline, clip),
            ("Delete", clip, None),
            ("Remove", clip, None),
        ]
        for name, obj, payload in delete_candidates:
            method = getattr(obj, name, None)
            if not callable(method):
                continue
            try:
                if payload is None:
                    result = method()
                else:
                    result = method(payload)
                logger.info("Replace clip debug: %s returned %s", name, result)
                break
            except Exception as exc:
                logger.debug("Replace clip debug: %s failed: %s", name, exc)

        # If deletion failed to remove the original clip, delete overlapping non-new clips.
        if start_frame is not None and end_frame is not None:
            try:
                overlap = self._find_items_overlapping_range(
                    track_type,
                    int(track_index),
                    int(start_frame),
                    int(end_frame),
                )
                if overlap:
                    remaining = []
                    for item in overlap:
                        if item is insert_item:
                            continue
                        media_path = self._timeline_item_media_path(item)
                        if media_path and str(media_path).lower() == str(new_path).lower():
                            continue
                        remaining.append(item)
                    if remaining:
                        logger.info(
                            "Replace clip debug: removing %d overlapping old items after insert",
                            len(remaining),
                        )
                        delete_method = getattr(timeline, "DeleteClips", None)
                        if callable(delete_method):
                            try:
                                delete_method(remaining)
                            except Exception as exc:
                                logger.info("Replace clip debug: overlap delete failed: %s", exc)
            except Exception as exc:
                logger.debug("Replace clip debug: overlap cleanup failed: %s", exc)

        # Best-effort ripple: shift later items left if the inserted clip is shorter.
        if ripple and start_frame is not None and end_frame is not None:
            try:
                new_start, new_end = self._get_item_range(insert_item)
                if new_start is None or new_end is None:
                    logger.info("Replace clip debug: ripple skipped (inserted item has no range)")
                    return True
                old_len = int(end_frame) - int(start_frame)
                new_len = int(new_end) - int(new_start)
                delta = old_len - new_len
                if delta <= 0:
                    logger.info("Replace clip debug: ripple not needed (delta=%s)", delta)
                    return True
                logger.info("Replace clip debug: ripple shifting later items left by %d frames", delta)
                self._ripple_shift_left(track_type, int(track_index), int(end_frame), delta)
            except Exception as exc:
                logger.info("Replace clip debug: ripple shift failed: %s", exc)
        return True

    def _find_items_overlapping_range(
        self,
        track_type: str,
        track_index: int,
        start_frame: int,
        end_frame: int,
    ) -> List[Any]:
        timeline = self.get_current_timeline()
        items = timeline.GetItemListInTrack(track_type, track_index) or []
        overlap: List[Any] = []
        for item in items:
            s, e = self._get_item_range(item)
            if s is None or e is None:
                continue
            if e < start_frame or s > end_frame:
                continue
            overlap.append(item)
        return overlap

    def _find_item_near_frame(
        self,
        track_type: str,
        track_index: int,
        frame: int,
        window: int | None = None,
    ) -> Optional[Any]:
        timeline = self.get_current_timeline()
        fps = self.get_timeline_fps()
        if window is None:
            window = max(2, int(round(fps * 0.25)))
        items = timeline.GetItemListInTrack(track_type, track_index) or []
        best = None
        best_delta = None
        for item in items:
            s, _e = self._get_item_range(item)
            if s is None:
                continue
            delta = abs(int(s) - int(frame))
            if delta <= window and (best_delta is None or delta < best_delta):
                best = item
                best_delta = delta
        return best

    def _snapshot_track_items(self, track_type: str) -> Dict[int, set[int]]:
        timeline = self.get_current_timeline()
        snapshot: Dict[int, set[int]] = {}
        track_count = int(timeline.GetTrackCount(track_type) or 0)
        for track_index in range(1, track_count + 1):
            items = timeline.GetItemListInTrack(track_type, track_index) or []
            snapshot[track_index] = {id(item) for item in items}
        return snapshot

    def _find_new_item_by_snapshot(
        self,
        track_type: str,
        snapshot: Dict[int, set[int]],
        media_path: Optional[str],
    ) -> Optional[tuple[int, Any]]:
        timeline = self.get_current_timeline()
        track_count = int(timeline.GetTrackCount(track_type) or 0)
        candidates: List[tuple[int, Any]] = []
        for track_index in range(1, track_count + 1):
            items = timeline.GetItemListInTrack(track_type, track_index) or []
            seen = snapshot.get(track_index, set())
            for item in items:
                if id(item) in seen:
                    continue
                candidates.append((track_index, item))
        if not candidates:
            return None
        if media_path:
            for track_index, item in candidates:
                path = self._timeline_item_media_path(item)
                if path and str(path).lower() == str(media_path).lower():
                    return track_index, item
        return candidates[0]

    def _timeline_item_media_path(self, item: Any) -> Optional[str]:
        prop = getattr(item, "GetClipProperty", None)
        if callable(prop):
            try:
                value = prop("File Path")
                if value:
                    return value
            except Exception:
                pass
            try:
                props = prop()
                if isinstance(props, dict):
                    path = props.get("File Path") or props.get("FilePath")
                    if path:
                        return path
            except Exception:
                pass
        pool_item = getattr(item, "GetMediaPoolItem", None)
        if callable(pool_item):
            try:
                media = pool_item()
                if media is not None:
                    prop = getattr(media, "GetClipProperty", None)
                    if callable(prop):
                        props = prop()
                        if isinstance(props, dict):
                            return props.get("File Path") or props.get("FilePath")
            except Exception:
                return None
        return None

    def get_item_media_path(self, item: Any) -> Optional[str]:
        """Return media file path for a timeline item."""
        return self._timeline_item_media_path(item)

    def _find_latest_item_by_media_path(
        self, track_type: str, track_index: int, path: str
    ) -> Optional[Any]:
        timeline = self.get_current_timeline()
        items = timeline.GetItemListInTrack(track_type, track_index) or []
        best = None
        best_start = None
        for item in items:
            media_path = self._timeline_item_media_path(item)
            if not media_path:
                continue
            if str(media_path).lower() != str(path).lower():
                continue
            s, _e = self._get_item_range(item)
            if s is None:
                continue
            if best is None or (best_start is not None and s > best_start) or best_start is None:
                best = item
                best_start = s
        return best

    def _set_item_start_frame(self, item: Any, frame: int) -> bool:
        for name in ("SetStart", "SetStartFrame", "SetStartTime"):
            method = getattr(item, name, None)
            if callable(method):
                try:
                    method(int(frame))
                    return True
                except Exception:
                    continue
        return False

    def _ripple_shift_left(self, track_type: str, track_index: int, after_frame: int, delta: int) -> None:
        timeline = self.get_current_timeline()
        items = timeline.GetItemListInTrack(track_type, track_index) or []
        movable = []
        for item in items:
            s, _e = self._get_item_range(item)
            if s is None:
                continue
            if s >= after_frame:
                movable.append((s, item))
        movable.sort(key=lambda x: x[0])
        shifted = 0
        for s, item in movable:
            if self._set_item_start_frame(item, int(s) - int(delta)):
                shifted += 1
        logger.info("Replace clip debug: ripple shifted %d items", shifted)

    def import_subtitles_into_timeline(self, srt_path: str) -> bool:
        """Import an SRT file into the current timeline subtitle track."""
        timeline = self.get_current_timeline()
        srt_path = str(Path(srt_path))
        if not Path(srt_path).exists():
            logger.warning("Subtitle import failed: file missing (%s)", srt_path)
            return False
        try:
            if int(timeline.GetTrackCount("subtitle") or 0) == 0:
                timeline.AddTrack("subtitle")
        except Exception as exc:
            logger.debug("Failed to ensure subtitle track: %s", exc)

        importer = getattr(timeline, "ImportSubtitles", None)
        if callable(importer):
            try:
                if importer(srt_path):
                    return True
            except Exception as exc:
                logger.warning("Timeline ImportSubtitles(path) failed: %s", exc)

        importer = getattr(timeline, "ImportSubtitlesFromFile", None)
        if callable(importer):
            try:
                if importer(srt_path):
                    return True
            except Exception as exc:
                logger.warning("Timeline ImportSubtitlesFromFile(path) failed: %s", exc)

        importer = getattr(timeline, "ImportIntoTimeline", None)
        if callable(importer):
            try:
                if importer(srt_path):
                    return True
            except Exception as exc:
                logger.warning("Timeline ImportIntoTimeline(path) failed: %s", exc)

            for payload in (
                {"filePath": srt_path},
                {"FilePath": srt_path},
                {"filepath": srt_path},
                {"path": srt_path},
                {"Path": srt_path},
                {"type": "subtitle", "filePath": srt_path},
                {"trackType": "subtitle", "filePath": srt_path},
                {"TrackType": "subtitle", "FilePath": srt_path},
                {"trackType": "subtitle", "trackIndex": 1, "filePath": srt_path},
            ):
                try:
                    if importer(payload):
                        return True
                except Exception as exc:
                    logger.debug("Timeline ImportIntoTimeline(%s) failed: %s", payload, exc)

        project = self.get_current_project()
        importer = getattr(project, "ImportSubtitles", None)
        if callable(importer):
            try:
                if importer(srt_path):
                    return True
            except Exception as exc:
                logger.warning("Project ImportSubtitles(path) failed: %s", exc)

        importer = getattr(project, "ImportIntoTimeline", None)
        if callable(importer):
            try:
                if importer(srt_path):
                    return True
            except Exception as exc:
                logger.warning("Project ImportIntoTimeline(path) failed: %s", exc)
        try:
            methods = [
                name
                for name in dir(timeline)
                if "import" in name.lower() or "subtitle" in name.lower()
            ]
            logger.debug("Timeline import methods: %s", methods)
        except Exception:
            pass
        return False


    @staticmethod
    def _subtitle_text(item: Any) -> Optional[str]:
        getter = getattr(item, "GetText", None)
        if callable(getter):
            try:
                text = getter()
                if text:
                    return str(text).strip()
            except Exception:
                pass
        for method_name in ("GetName",):
            getter = getattr(item, method_name, None)
            if callable(getter):
                try:
                    text = getter()
                    if text:
                        return str(text).strip()
                except Exception:
                    pass
        getter = getattr(item, "GetProperty", None)
        if callable(getter):
            for key in ("Text", "text", "Name", "name"):
                try:
                    value = getter(key)
                    if value:
                        return str(value).strip()
                except Exception:
                    continue
        getter = getattr(item, "GetClipProperty", None)
        if callable(getter):
            for key in ("Text", "text", "Name", "name"):
                try:
                    value = getter(key)
                    if value:
                        return str(value).strip()
                except Exception:
                    continue
        return None

    @staticmethod
    def _item_time_range_seconds(item: Any, fps: float) -> tuple[float, float]:
        start, end = ResolveAPI._get_item_range(item)
        if start is None or end is None or fps <= 0:
            return 0.0, 0.0
        return float(start) / float(fps), float(end) / float(fps)

    def get_clip_at_playhead(self) -> Optional[Any]:
        """Return the timeline clip under the playhead if available."""
        timeline = self.get_current_timeline()
        try:
            current_item = timeline.GetCurrentVideoItem()
            if current_item:
                return current_item
        except Exception:
            pass

        frame = self._get_playhead_frame(timeline)
        if frame is None:
            return None

        track_count = int(timeline.GetTrackCount("video") or 0)
        for track_index in range(1, track_count + 1):
            items = timeline.GetItemListInTrack("video", track_index) or []
            for item in items:
                start, end = self._get_item_range(item)
                if start is None or end is None:
                    continue
                if start <= frame <= end:
                    return item
        return None

    def _get_playhead_frame(self, timeline: Any) -> Optional[int]:
        try:
            timecode = timeline.GetCurrentTimecode()
        except Exception:
            return None
        fps = self.get_timeline_fps()
        return self._timecode_to_frames(timecode, fps)

    @staticmethod
    def _timecode_to_frames(timecode: str, fps: float) -> Optional[int]:
        try:
            parts = [int(p) for p in str(timecode).split(":")]
            if len(parts) != 4:
                return None
            hours, minutes, seconds, frames = parts
            return int(round(((hours * 3600 + minutes * 60 + seconds) * fps) + frames))
        except Exception as exc:
            logger.debug("Failed to parse timecode %s: %s", timecode, exc)
            return None

    @staticmethod
    def _get_item_range(item: Any) -> tuple[Optional[int], Optional[int]]:
        for start_name, end_name in (("GetStart", "GetEnd"), ("GetStartFrame", "GetEndFrame")):
            try:
                start = getattr(item, start_name)()
                end = getattr(item, end_name)()
                if start is not None and end is not None:
                    return int(start), int(end)
            except Exception:
                continue
        return None, None

    def get_media_pool(self) -> Any:
        """Return the project media pool."""
        project = self.get_current_project()
        return project.GetMediaPool()

    def export_clip_audio(
        self,
        clip: Any,
        output_path: str,
        start_sec: float | None = None,
        end_sec: float | None = None,
    ) -> str:
        """Export clip audio to a WAV file using ffmpeg."""
        if clip is None:
            raise RuntimeError("Selected clip is missing.")
        source = self._timeline_item_media_path(clip)
        if not source:
            raise RuntimeError("Selected clip has no file path.")
        output = str(Path(output_path))
        cmd = ["ffmpeg", "-y", "-i", source]
        if start_sec is not None:
            cmd.extend(["-ss", f"{float(start_sec):.3f}"])
        if end_sec is not None and start_sec is not None:
            duration = float(end_sec) - float(start_sec)
            if duration > 0:
                cmd.extend(["-t", f"{duration:.3f}"])
        elif end_sec is not None:
            cmd.extend(["-to", f"{float(end_sec):.3f}"])
        cmd.extend(["-vn", "-ac", "1", "-ar", "48000", output])
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                encoding="utf-8",
                errors="ignore",
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Command timed out after 300s: {cmd[0]}") from exc
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()}")
        return output

    def add_beat_markers(
        self,
        beat_times: List[float],
        marker_color: str = "blue",
        marker_name: str = "Beat",
        base_frame: Optional[int] = None,
    ) -> bool:
        """Add beat markers to the current timeline."""
        self._ensure_main_thread("add_beat_markers")
        if not beat_times:
            return False
        timeline = self.get_current_timeline()
        fps = float(self.get_timeline_fps())
        start_frame = int(base_frame) if base_frame is not None else int(self.get_timeline_start_frame())
        added = 0
        for beat in beat_times:
            try:
                frame = int(round(float(beat) * fps)) + start_frame
            except Exception:
                continue
            if self._add_timeline_marker(timeline, frame, marker_color, marker_name, ""):
                added += 1
        logger.info("Beat markers added: %s", added)
        return added > 0

    def add_downbeat_markers(
        self,
        downbeat_times: List[float],
        marker_color: str = "red",
        marker_name: str = "Downbeat",
        base_frame: Optional[int] = None,
    ) -> bool:
        """Add downbeat markers to the current timeline."""
        self._ensure_main_thread("add_downbeat_markers")
        if not downbeat_times:
            return False
        timeline = self.get_current_timeline()
        fps = float(self.get_timeline_fps())
        start_frame = int(base_frame) if base_frame is not None else int(self.get_timeline_start_frame())
        added = 0
        for beat in downbeat_times:
            try:
                frame = int(round(float(beat) * fps)) + start_frame
            except Exception:
                continue
            if self._add_timeline_marker(timeline, frame, marker_color, marker_name, ""):
                added += 1
        logger.info("Downbeat markers added: %s", added)
        return added > 0

    @staticmethod
    def _add_timeline_marker(
        timeline: Any,
        frame: int,
        color: str,
        name: str,
        note: str,
        duration: int = 1,
    ) -> bool:
        global _LOGGED_MARKER_MISSING
        global _LOGGED_MARKER_FAILED
        add_marker = getattr(timeline, "AddMarker", None)
        if not callable(add_marker):
            if not _LOGGED_MARKER_MISSING:
                logger.warning("Timeline AddMarker not available; beat markers disabled.")
                _LOGGED_MARKER_MISSING = True
            return False
        color_name = ResolveAPI._normalize_marker_color(color)
        try:
            result = add_marker(
                int(frame),
                str(color_name),
                str(name),
                str(note),
                int(duration),
                {},
            )
            if bool(result) or result is None:
                return True
            if not _LOGGED_MARKER_FAILED:
                logger.warning(
                    "Timeline AddMarker returned False (color=%s frame=%s).",
                    color_name,
                    frame,
                )
                _LOGGED_MARKER_FAILED = True
            return False
        except Exception:
            return False

    @staticmethod
    def _normalize_marker_color(color: str) -> str:
        value = str(color or "").strip().lower()
        palette = {
            "blue": "Blue",
            "cyan": "Cyan",
            "green": "Green",
            "yellow": "Yellow",
            "red": "Red",
            "pink": "Pink",
            "purple": "Purple",
            "fuchsia": "Fuchsia",
            "rose": "Rose",
            "lavender": "Lavender",
            "sky": "Sky",
            "mint": "Mint",
            "lime": "Lime",
            "sand": "Sand",
            "cocoa": "Cocoa",
            "cream": "Cream",
        }
        return palette.get(value, "Blue")

    def insert_clip_at_position(
        self, track_type: str, track_index: int, clip: Any, timeline_position: int
    ) -> Optional[Any]:
        """Insert a media pool clip at a timeline position (frames)."""
        self.ensure_track(track_type, track_index)
        timeline = self.get_current_timeline()
        media_pool = self.get_media_pool()
        fps = self.get_timeline_fps()
        duration_frames = self._get_media_item_duration_frames(clip, fps)
        record_frame = int(timeline_position)
        end_frame = max(0, duration_frames - 1) if duration_frames else None
        pre_snapshot = self._snapshot_track_items(track_type)
        media_path = None
        try:
            props = clip.GetClipProperty()
            if isinstance(props, dict):
                media_path = props.get("File Path") or props.get("FilePath")
        except Exception:
            media_path = None

        logger.info(
            "Insert clip debug: track=%s index=%s frame=%s",
            track_type,
            track_index,
            timeline_position,
        )

        insert_method = getattr(media_pool, "InsertClips", None)
        if callable(insert_method):
            try:
                insert_method([clip], track_type, track_index, timeline_position)
                logger.info("Insert clip debug: media_pool.InsertClips succeeded")
                found = self._find_new_item_by_snapshot(track_type, pre_snapshot, media_path)
                if found:
                    found_track, inserted = found
                    if found_track != track_index:
                        logger.info(
                            "Insert clip debug: inserted on track %s (requested %s)",
                            found_track,
                            track_index,
                        )
                    return inserted
                return None
            except Exception as exc:
                logger.info("Insert clip debug: media_pool.InsertClips failed: %s", exc)

        insert_method = getattr(timeline, "InsertClips", None)
        if callable(insert_method):
            try:
                insert_method([clip], track_index, timeline_position)
                logger.info("Insert clip debug: timeline.InsertClips succeeded")
                found = self._find_new_item_by_snapshot(track_type, pre_snapshot, media_path)
                if found:
                    found_track, inserted = found
                    if found_track != track_index:
                        logger.info(
                            "Insert clip debug: inserted on track %s (requested %s)",
                            found_track,
                            track_index,
                        )
                    return inserted
                return None
            except Exception as exc:
                logger.info("Insert clip debug: timeline.InsertClips failed: %s", exc)

        append_method = getattr(media_pool, "AppendToTimeline", None)
        if callable(append_method):
            # Some Resolve builds accept dict payloads with startFrame/track info.
            payloads = [
                {"mediaPoolItem": clip, "startFrame": int(timeline_position)},
                {"mediaPoolItem": clip, "startFrame": int(timeline_position), "trackIndex": int(track_index)},
                {"mediaPoolItem": clip, "startFrame": int(timeline_position), "trackIndex": int(track_index), "trackType": track_type},
                {"mediaPoolItem": clip, "startFrame": int(timeline_position), "trackType": track_type},
            ]
            if end_frame is not None:
                payloads = [
                    {
                        "mediaPoolItem": clip,
                        "startFrame": 0,
                        "endFrame": end_frame,
                        "recordFrame": record_frame,
                        "trackIndex": int(track_index),
                        "trackType": track_type,
                    },
                    {
                        "mediaPoolItem": clip,
                        "startFrame": 0,
                        "endFrame": end_frame,
                        "recordFrame": record_frame,
                        "trackIndex": int(track_index),
                    },
                    {
                        "mediaPoolItem": clip,
                        "startFrame": 0,
                        "endFrame": end_frame,
                        "recordFrame": record_frame,
                    },
                ] + payloads
            for payload in payloads:
                try:
                    append_method([payload])
                    logger.info("Insert clip debug: AppendToTimeline payload succeeded: %s", payload)
                    found = self._find_new_item_by_snapshot(track_type, pre_snapshot, media_path)
                    if found:
                        found_track, inserted = found
                        if found_track != track_index:
                            logger.info(
                                "Insert clip debug: inserted on track %s (requested %s)",
                                found_track,
                                track_index,
                            )
                        return inserted
                    return None
                except Exception as exc:
                    logger.debug("Insert clip debug: AppendToTimeline payload failed: %s", exc)

            set_frame = getattr(timeline, "SetCurrentFrame", None)
            set_timecode = getattr(timeline, "SetCurrentTimecode", None)
            get_frame = getattr(timeline, "GetCurrentFrame", None)
            get_timecode = getattr(timeline, "GetCurrentTimecode", None)
            prev_frame = None
            prev_timecode = None
            if callable(get_frame):
                try:
                    prev_frame = int(get_frame())
                except Exception:
                    prev_frame = None
            if prev_frame is None and callable(get_timecode):
                try:
                    prev_timecode = str(get_timecode())
                except Exception:
                    prev_timecode = None
            if callable(set_frame):
                try:
                    set_frame(int(timeline_position))
                except Exception as exc:
                    logger.info("Insert clip debug: SetCurrentFrame failed: %s", exc)
            elif callable(set_timecode):
                try:
                    set_timecode(self._frames_to_timecode(int(timeline_position), fps))
                except Exception as exc:
                    logger.info("Insert clip debug: SetCurrentTimecode failed: %s", exc)
            try:
                append_method([clip])
                logger.info("Insert clip debug: media_pool.AppendToTimeline succeeded")
                found = self._find_new_item_by_snapshot(track_type, pre_snapshot, media_path)
                if found:
                    found_track, inserted = found
                else:
                    inserted = None
                    found_track = None
                if inserted is None:
                    logger.info("Insert clip debug: appended, but could not locate inserted item")
                    return None
                if self._set_item_start_frame(inserted, int(timeline_position)):
                    logger.info("Insert clip debug: repositioned appended item to frame=%s", timeline_position)
                    if found_track and found_track != track_index:
                        logger.info(
                            "Insert clip debug: appended on track %s (requested %s)",
                            found_track,
                            track_index,
                        )
                    return inserted
                logger.info("Insert clip debug: appended item could not be repositioned")
                return None
            except Exception as exc:
                logger.info("Insert clip debug: media_pool.AppendToTimeline failed: %s", exc)
            finally:
                if prev_frame is not None and callable(set_frame):
                    try:
                        set_frame(prev_frame)
                    except Exception:
                        pass
                elif prev_timecode is not None and callable(set_timecode):
                    try:
                        set_timecode(prev_timecode)
                    except Exception:
                        pass

        logger.info("Insert clip debug: no supported insert method")
        return None

    def insert_clip_at_position_with_range(
        self,
        track_type: str,
        track_index: int,
        clip: Any,
        timeline_position: int,
        source_start: int,
        source_end: int,
    ) -> Optional[Any]:
        """Insert a media pool clip with source in/out frames at a timeline position."""
        self.ensure_track(track_type, track_index)
        media_pool = self.get_media_pool()
        record_frame = int(timeline_position)
        start_frame = max(0, int(source_start))
        end_frame = max(start_frame + 1, int(source_end))

        logger.info(
            "Insert clip range debug: track=%s index=%s record=%s source=%s-%s",
            track_type,
            track_index,
            record_frame,
            start_frame,
            end_frame,
        )

        append_method = getattr(media_pool, "AppendToTimeline", None)
        if callable(append_method):
            timeline = self.get_current_timeline()
            pre_items = timeline.GetItemListInTrack(track_type, int(track_index)) or []
            pre_ids = {id(item) for item in pre_items}
            payload = {
                "mediaPoolItem": clip,
                "startFrame": start_frame,
                "endFrame": end_frame,
                "recordFrame": record_frame,
                "trackIndex": int(track_index),
                "trackType": track_type,
            }
            try:
                append_method([payload])
                logger.info(
                    "Insert clip range debug: AppendToTimeline payload succeeded: %s",
                    payload,
                )
                post_items = timeline.GetItemListInTrack(track_type, int(track_index)) or []
                new_items = [item for item in post_items if id(item) not in pre_ids]
                inserted = None
                if new_items:
                    if len(new_items) == 1:
                        inserted = new_items[0]
                    else:
                        for item in new_items:
                            item_path = self._timeline_item_media_path(item)
                            clip_path = None
                            try:
                                prop = getattr(clip, "GetClipProperty", None)
                                if callable(prop):
                                    props = prop() or {}
                                    if isinstance(props, dict):
                                        clip_path = props.get("File Path") or props.get("FilePath")
                            except Exception:
                                clip_path = None
                            if clip_path and item_path and str(item_path).lower() != str(clip_path).lower():
                                continue
                            inserted = item
                            break
                if inserted is None:
                    inserted = self._find_item_near_frame(
                        track_type, track_index, int(record_frame)
                    )
                    if inserted is not None and id(inserted) in pre_ids:
                        inserted = None
                if inserted is None:
                    track_count = int(timeline.GetTrackCount(track_type) or 0)
                    for idx in range(1, track_count + 1):
                        if idx == int(track_index):
                            continue
                        post_items = timeline.GetItemListInTrack(track_type, idx) or []
                        new_items = [item for item in post_items if id(item) not in pre_ids]
                        if not new_items:
                            continue
                        inserted = new_items[0]
                        logger.info(
                            "Insert clip range debug: inserted on track %s (requested %s)",
                            idx,
                            track_index,
                        )
                        break
                if inserted is None:
                    logger.info(
                        "Insert clip range debug: append succeeded but timeline item not located"
                    )
                    return None
                return inserted
            except Exception as exc:
                logger.info("Insert clip range debug: AppendToTimeline failed: %s", exc)

        logger.info("Insert clip range debug: no supported insert method")
        return None

    def add_synced_clips_to_timeline(
        self,
        reference_path: str,
        sync_results: Dict[Path, Dict[str, Any]],
        start_frame: int = 0,
    ) -> bool:
        """Place synced clips on separate tracks with offset compensation."""
        self._ensure_main_thread("add_synced_clips_to_timeline")
        try:
            timeline = self.get_current_timeline()
        except Exception as exc:
            logger.error("No active timeline: %s", exc)
            return False

        media_pool = self.get_media_pool()
        if not media_pool:
            logger.error("Cannot access media pool")
            return False

        fps = self.get_timeline_fps()
        target_paths = list(sync_results.keys())
        all_paths = [reference_path] + [str(p) for p in target_paths]

        try:
            imported_clips = media_pool.ImportMedia(all_paths)
        except Exception as exc:
            logger.error("Failed to import media: %s", exc)
            return False

        if not imported_clips:
            logger.error("No clips imported")
            return False

        if start_frame == 0:
            playhead = self._get_playhead_frame(timeline)
            if playhead is not None:
                start_frame = playhead

        ref_clip = imported_clips[0]
        if not self._add_clip_to_track(
            timeline,
            media_pool,
            ref_clip,
            track_index=1,
            track_type="video",
            start_frame=int(start_frame),
        ):
            logger.error("Failed to place reference clip")
            return False

        logger.info(
            "Placed reference: %s at frame %d",
            Path(reference_path).name,
            int(start_frame),
        )

        for idx, target_path in enumerate(target_paths, start=2):
            clip_index = idx - 1
            if clip_index >= len(imported_clips):
                logger.warning("Missing imported clip for %s", target_path)
                continue
            target_clip = imported_clips[clip_index]
            result = sync_results.get(target_path, {})
            offset_seconds = float(result.get("offset", 0.0))
            confidence = float(result.get("confidence", 0.0))
            offset_frames = int(-offset_seconds * fps)
            target_start_frame = int(start_frame) + int(offset_frames)
            success = self._add_clip_to_track(
                timeline,
                media_pool,
                target_clip,
                track_index=idx,
                track_type="video",
                start_frame=target_start_frame,
            )
            if success:
                logger.info(
                    "Placed target: %s at frame %d (offset=%.2fs, confidence=%.2f)",
                    Path(target_path).name,
                    target_start_frame,
                    offset_seconds,
                    confidence,
                )
            else:
                logger.warning("Failed to place: %s", Path(target_path).name)

        return True

    def _add_clip_to_track(
        self,
        timeline: Any,
        media_pool: Any,
        clip: Any,
        track_index: int,
        track_type: str = "video",
        start_frame: int = 0,
    ) -> bool:
        """Helper for placing a clip on a specific track."""
        _ = timeline
        _ = media_pool
        try:
            self.ensure_track(track_type, int(track_index))
        except Exception as exc:
            logger.error("Failed to ensure track %s %d: %s", track_type, track_index, exc)
            return False
        inserted = self.insert_clip_at_position(
            track_type,
            int(track_index),
            clip,
            int(start_frame),
        )
        return inserted is not None

    def get_timeline_fps(self) -> float:
        """Return timeline FPS or 24 if missing."""
        project = self.get_current_project()
        try:
            fps = float(project.GetSetting("timelineFrameRate"))
        except Exception:
            fps = 24.0
        return fps

    def get_timeline_markers(self) -> Dict[int, Dict[str, Any]]:
        """Return timeline markers keyed by frame (best effort)."""
        self._ensure_main_thread("get_timeline_markers")
        timeline = self.get_current_timeline()
        getter = getattr(timeline, "GetMarkers", None)
        if callable(getter):
            try:
                markers = getter()
                if isinstance(markers, dict):
                    result = markers
                    timeline_start = int(self.get_timeline_start_frame() or 0)
                    if result and timeline_start > 0:
                        try:
                            max_frame = max(int(key) for key in result.keys())
                            min_frame = min(int(key) for key in result.keys())
                        except Exception:
                            max_frame = None
                            min_frame = None
                        if max_frame is not None and max_frame < timeline_start:
                            logger.info(
                                "Beat markers appear relative; offsetting by timeline_start=%s (min=%s max=%s).",
                                timeline_start,
                                min_frame,
                                max_frame,
                            )
                            result = {int(frame) + timeline_start: meta for frame, meta in result.items()}
                    return result
                if isinstance(markers, (list, tuple)):
                    result: Dict[int, Dict[str, Any]] = {}
                    for item in markers:
                        frame = self._extract_marker_frame(item)
                        if frame is None:
                            continue
                        meta: Dict[str, Any] = {}
                        if isinstance(item, dict):
                            meta = item
                        else:
                            for key in ("name", "Name", "color", "Color", "note", "Note"):
                                value = getattr(item, key, None)
                                if value is not None:
                                    meta[key] = value
                        result[int(frame)] = meta
                    timeline_start = int(self.get_timeline_start_frame() or 0)
                    if result and timeline_start > 0:
                        try:
                            max_frame = max(int(key) for key in result.keys())
                            min_frame = min(int(key) for key in result.keys())
                        except Exception:
                            max_frame = None
                            min_frame = None
                        if max_frame is not None and max_frame < timeline_start:
                            logger.info(
                                "Beat markers appear relative; offsetting by timeline_start=%s (min=%s max=%s).",
                                timeline_start,
                                min_frame,
                                max_frame,
                            )
                            result = {int(frame) + timeline_start: meta for frame, meta in result.items()}
                    return result
            except Exception:
                return {}
        return {}

    def get_first_marker_frame(self) -> Optional[int]:
        """Return the earliest marker frame on the timeline (best effort)."""
        self._ensure_main_thread("get_first_marker_frame")
        markers = self.get_timeline_markers()
        if not markers:
            return None
        frames: List[int] = []
        for key in markers.keys():
            try:
                frames.append(int(key))
            except Exception:
                continue
        if not frames:
            return None
        return int(min(frames))

    @staticmethod
    def _extract_marker_frame(item: Any) -> Optional[int]:
        if item is None:
            return None
        if isinstance(item, dict):
            for key in ("frameId", "frame", "FrameId", "Frame", "frame_id"):
                if key in item:
                    try:
                        return int(item[key])
                    except Exception:
                        continue
        if isinstance(item, (list, tuple)) and item:
            try:
                return int(item[0])
            except Exception:
                pass
        for name in ("GetFrameId", "GetFrame", "frame", "frameId"):
            value = getattr(item, name, None)
            if callable(value):
                try:
                    return int(value())
                except Exception:
                    continue
            if value is not None:
                try:
                    return int(value)
                except Exception:
                    continue
        return None

    def get_timeline_start_frame(self) -> int:
        """Return the timeline start frame offset (timecode-based) if available."""
        timeline = self.get_current_timeline()
        fps = self.get_timeline_fps()
        start_frame = None
        for getter in ("GetStartFrame", "GetStartTC", "GetStartTimecode", "GetStartTimeCode"):
            method = getattr(timeline, getter, None)
            if callable(method):
                try:
                    value = method()
                except Exception:
                    value = None
                if isinstance(value, (int, float)):
                    start_frame = int(value)
                    break
                if value:
                    start_frame = self._timecode_to_frames(str(value), fps)
                    if start_frame is not None:
                        break
        if start_frame is None:
            project = self.get_current_project()
            for key in ("timelineStartTimecode", "timelineStartTC"):
                try:
                    value = project.GetSetting(key)
                except Exception:
                    value = None
                if value:
                    start_frame = self._timecode_to_frames(str(value), fps)
                    if start_frame is not None:
                        break
        return int(start_frame) if start_frame is not None else 0

    @staticmethod
    def _frames_to_timecode(frame: int, fps: float) -> str:
        fps_int = max(1, int(round(fps)))
        total_seconds = frame // fps_int
        frames = frame % fps_int
        seconds = total_seconds % 60
        minutes = (total_seconds // 60) % 60
        hours = total_seconds // 3600
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"

    def _get_media_item_duration_frames(self, clip: Any, fps: float) -> int:
        props = {}
        prop_method = getattr(clip, "GetClipProperty", None)
        if callable(prop_method):
            try:
                props = prop_method() or {}
            except Exception:
                props = {}
        duration = props.get("Duration") or props.get("Frames") or props.get("FrameCount")
        if duration:
            try:
                return int(float(duration))
            except Exception:
                pass
            try:
                parts = [int(p) for p in str(duration).split(":")]
                if len(parts) == 4:
                    hours, minutes, seconds, frames = parts
                    fps_int = max(1, int(round(fps)))
                    return int((hours * 3600 + minutes * 60 + seconds) * fps_int + frames)
            except Exception:
                pass
        media_path = None
        if props:
            media_path = props.get("File Path") or props.get("FilePath")
        if not media_path:
            try:
                media_path = clip.GetClipProperty("File Path")
            except Exception:
                media_path = None
        if media_path:
            duration_sec = self._ffprobe_duration_seconds(str(media_path))
            if duration_sec:
                return int(round(duration_sec * fps))
        return 0

    @staticmethod
    def _parse_fps_value(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            pass
        text = str(value).strip().lower()
        if not text:
            return None
        for token in text.replace(",", ".").split():
            try:
                return float(token)
            except Exception:
                continue
        return None

    def _get_media_item_fps(self, clip: Any, fallback_fps: float) -> float:
        props = {}
        prop_method = getattr(clip, "GetClipProperty", None)
        if callable(prop_method):
            try:
                props = prop_method() or {}
            except Exception:
                props = {}
        for key in ("FPS", "FrameRate", "Frame Rate"):
            value = None
            if props:
                value = props.get(key)
            if value is None and callable(prop_method):
                try:
                    value = clip.GetClipProperty(key)
                except Exception:
                    value = None
            fps = self._parse_fps_value(value)
            if fps:
                return float(fps)
        return float(fallback_fps)

    @staticmethod
    def _ffprobe_duration_seconds(path: str) -> float | None:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                encoding="utf-8",
                errors="ignore",
            )
        except subprocess.TimeoutExpired:
            return None
        if proc.returncode != 0:
            return None
        try:
            return float(proc.stdout.strip())
        except Exception:
            return None

    def ensure_track(self, track_type: str, track_index: int) -> None:
        """Ensure a timeline track index exists."""
        timeline = self.get_current_timeline()
        track_count = int(timeline.GetTrackCount(track_type) or 0)
        while track_count < track_index:
            timeline.AddTrack(track_type)
            track_count += 1

    def import_media_if_needed(self, path: str) -> Optional[Any]:
        """Import media if missing and return the media pool clip."""
        media_pool = self.get_media_pool()
        root = media_pool.GetRootFolder()
        for clip in root.GetClipList() or []:
            if clip.GetClipProperty("File Path") == path:
                return clip
        imported = media_pool.ImportMedia([path])
        return imported[0] if imported else None

    def add_media_to_bin(self, path: str, bin_name: str = "VideoForge Rendered") -> bool:
        """Import media (if needed) and place it into a named bin (best effort)."""
        media_pool = self.get_media_pool()
        root = media_pool.GetRootFolder()
        clip = self.import_media_if_needed(path)
        if clip is None:
            logger.info("Bin add failed: import failed for %s", path)
            return False

        folder = self._ensure_subfolder(root, bin_name)
        if folder is None:
            logger.info("Bin add skipped: could not ensure folder %s", bin_name)
            return False

        move = getattr(media_pool, "MoveClips", None)
        if callable(move):
            try:
                move([clip], folder)
                logger.info("Bin add: moved clip into %s", bin_name)
                return True
            except Exception as exc:
                logger.info("Bin add: MoveClips failed: %s", exc)

        set_current = getattr(media_pool, "SetCurrentFolder", None)
        if callable(set_current):
            try:
                set_current(folder)
            except Exception:
                pass
        logger.info("Bin add: could not move clip; folder ensured for manual access")
        return True

    def insert_rendered_above_clip(
        self,
        timeline_clip: Any,
        rendered_path: str,
        include_audio: bool = True,
    ) -> bool:
        """Insert a rendered clip above the selected clip's track at the same timeline position."""
        media_item = self.import_media_if_needed(rendered_path)
        if media_item is None:
            logger.info("Insert above failed: import failed for %s", rendered_path)
            return False

        start_frame, end_frame = self._get_item_range(timeline_clip)
        if start_frame is None:
            logger.info("Insert above failed: missing timeline start frame")
            return False

        video_track_index = self._safe_call(timeline_clip, "GetTrackIndex") or 1
        target_video_index = int(video_track_index) + 1
        logger.info(
            "Insert above debug: video track %s -> %s at frame %s",
            video_track_index,
            target_video_index,
            start_frame,
        )
        inserted_video = self.insert_clip_at_position("video", target_video_index, media_item, int(start_frame))
        ok = inserted_video is not None

        if include_audio:
            audio_track_index = self._find_linked_audio_track_index(timeline_clip)
            if audio_track_index is not None:
                target_audio_index = int(audio_track_index) + 1
                logger.info(
                    "Insert above debug: audio track %s -> %s at frame %s",
                    audio_track_index,
                    target_audio_index,
                    start_frame,
                )
                inserted_audio = self.insert_clip_at_position(
                    "audio",
                    target_audio_index,
                    media_item,
                    int(start_frame),
                )
                ok = ok and (inserted_audio is not None)
            else:
                logger.info("Insert above debug: no linked audio track detected; skipping audio insert")

        return ok

    def insert_overlapped_segments(
        self,
        timeline_clip: Any,
        segments: List[Dict[str, float]],
        overlap_frames: int = 15,
        overlap_jitter: int = 0,
        jitter_seed: int | None = None,
        include_audio: bool = True,
    ) -> bool:
        """Insert segments on alternating tracks with frame overlap."""
        if not timeline_clip:
            return False
        if not segments:
            logger.info("Overlap insert skipped: no segments")
            return False

        media_path = self._timeline_item_media_path(timeline_clip)
        if not media_path:
            logger.info("Overlap insert failed: missing media path")
            return False
        media_item = self.import_media_if_needed(str(media_path))
        if media_item is None:
            logger.info("Overlap insert failed: import failed for %s", media_path)
            return False

        timeline_start, _timeline_end = self._get_item_range(timeline_clip)
        if timeline_start is None:
            logger.info("Overlap insert failed: missing timeline start frame")
            return False

        fps = self.get_timeline_fps()
        overlap_frames = max(0, int(overlap_frames))
        overlap_jitter = max(0, int(overlap_jitter))
        rng = random.Random(jitter_seed) if overlap_jitter else None

        base_video_index = self._safe_call(timeline_clip, "GetTrackIndex") or 1
        video_track_a = int(base_video_index) + 1
        video_track_b = int(base_video_index) + 2
        audio_track_a = None
        audio_track_b = None
        if include_audio:
            audio_index = self._find_linked_audio_track_index(timeline_clip)
            if audio_index is not None:
                audio_track_a = int(audio_index) + 1
                audio_track_b = int(audio_index) + 2

        logger.info(
            "Overlap insert: video tracks %s/%s audio tracks %s/%s overlap=%s frames",
            video_track_a,
            video_track_b,
            audio_track_a,
            audio_track_b,
            overlap_frames,
        )
        if overlap_jitter:
            logger.info(
                "Overlap insert jitter: +/- %s frames (seed=%s)",
                overlap_jitter,
                jitter_seed,
            )

        cursor = int(timeline_start)
        inserted = 0
        dropped = 0

        for idx, seg in enumerate(segments):
            t0 = float(seg.get("t0", 0.0))
            t1 = float(seg.get("t1", 0.0))
            if t1 <= t0:
                dropped += 1
                continue
            start_frame = int(round(t0 * fps))
            end_frame = int(round(t1 * fps))
            duration = end_frame - start_frame
            if duration <= 0:
                dropped += 1
                continue
            overlap_for_segment = overlap_frames
            if overlap_jitter:
                jitter = rng.randint(-overlap_jitter, overlap_jitter) if rng else 0
                overlap_for_segment = max(0, overlap_frames + jitter)
            if overlap_for_segment and duration <= overlap_for_segment:
                dropped += 1
                continue

            track_video = video_track_a if (idx % 2 == 0) else video_track_b
            track_audio = audio_track_a if (idx % 2 == 0) else audio_track_b

            video_item = self.insert_clip_at_position_with_range(
                "video",
                track_video,
                media_item,
                cursor,
                start_frame,
                end_frame,
            )
            ok = video_item is not None
            if include_audio and track_audio is not None:
                audio_item = self.insert_clip_at_position_with_range(
                    "audio",
                    track_audio,
                    media_item,
                    cursor,
                    start_frame,
                    end_frame,
                )
                ok = ok and (audio_item is not None)

            if ok:
                inserted += 1

            cursor += max(1, duration - overlap_for_segment)

        logger.info(
            "Overlap insert complete: inserted=%d dropped=%d", inserted, dropped
        )
        return inserted > 0

    def _find_linked_audio_track_index(self, video_item: Any) -> Optional[int]:
        """Try to find an audio timeline item that matches the given video item."""
        timeline = self.get_current_timeline()
        start_frame, end_frame = self._get_item_range(video_item)
        if start_frame is None or end_frame is None:
            return None
        video_path = self._timeline_item_media_path(video_item)
        track_count = int(timeline.GetTrackCount("audio") or 0)
        for track_index in range(1, track_count + 1):
            items = timeline.GetItemListInTrack("audio", track_index) or []
            for item in items:
                s, e = self._get_item_range(item)
                if s is None or e is None:
                    continue
                if abs(s - start_frame) > 2 or abs(e - end_frame) > 2:
                    continue
                if video_path:
                    audio_path = self._timeline_item_media_path(item)
                    if audio_path and str(audio_path).lower() != str(video_path).lower():
                        continue
                return int(track_index)
        return None

    def _ensure_subfolder(self, parent: Any, name: str) -> Optional[Any]:
        media_pool = self.get_media_pool()
        list_method = getattr(parent, "GetSubFolderList", None)
        if callable(list_method):
            try:
                for folder in list_method() or []:
                    if getattr(folder, "GetName", lambda: "")() == name:
                        return folder
            except Exception:
                pass
        add_method = getattr(media_pool, "AddSubFolder", None)
        if callable(add_method):
            try:
                folder = add_method(parent, name)
                if folder is not None:
                    return folder
            except Exception as exc:
                logger.info("Bin add: AddSubFolder failed: %s", exc)
        return None
