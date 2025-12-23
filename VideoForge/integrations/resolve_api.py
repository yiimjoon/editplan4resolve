import os
import subprocess
import logging
import threading
from pathlib import Path
from typing import Any, List, Optional, Iterable, Dict

from VideoForge.integrations.resolve_python import _get_resolve

logger = logging.getLogger(__name__)


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

    def _timeline_item_media_path(self, item: Any) -> Optional[str]:
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

    def export_clip_audio(self, clip: Any, output_path: str) -> str:
        """Export clip audio to a WAV file using ffmpeg."""
        source = clip.GetClipProperty("File Path")
        if not source:
            raise RuntimeError("Selected clip has no file path.")
        output = str(Path(output_path))
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            source,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "48000",
            output,
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
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Command timed out after 300s: {cmd[0]}") from exc
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()}")
        return output

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
                inserted = self._find_item_near_frame(
                    track_type, track_index, int(timeline_position)
                )
                if inserted is not None:
                    return inserted
                return None
            except Exception as exc:
                logger.info("Insert clip debug: media_pool.InsertClips failed: %s", exc)

        insert_method = getattr(timeline, "InsertClips", None)
        if callable(insert_method):
            try:
                insert_method([clip], track_index, timeline_position)
                logger.info("Insert clip debug: timeline.InsertClips succeeded")
                inserted = self._find_item_near_frame(
                    track_type, track_index, int(timeline_position)
                )
                if inserted is not None:
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
                    inserted = self._find_item_near_frame(
                        track_type, track_index, int(timeline_position)
                    )
                    if inserted is not None:
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
                media_path = None
                try:
                    media_path = clip.GetClipProperty("File Path")
                except Exception:
                    media_path = None
                if not media_path:
                    return None
                inserted = self._find_latest_item_by_media_path(track_type, track_index, str(media_path))
                if inserted is None:
                    logger.info("Insert clip debug: appended, but could not locate inserted item for reposition")
                    return None
                if self._set_item_start_frame(inserted, int(timeline_position)):
                    logger.info("Insert clip debug: repositioned appended item to frame=%s", timeline_position)
                    return self._find_item_near_frame(
                        track_type, track_index, int(timeline_position)
                    )
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

    def get_timeline_fps(self) -> float:
        """Return timeline FPS or 24 if missing."""
        project = self.get_current_project()
        try:
            fps = float(project.GetSetting("timelineFrameRate"))
        except Exception:
            fps = 24.0
        return fps

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
