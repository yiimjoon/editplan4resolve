import subprocess
import logging
from pathlib import Path
from typing import Any, List, Optional

from VideoForge.integrations.resolve_python import _get_resolve

logger = logging.getLogger(__name__)


class ResolveAPI:
    """Convenience wrapper for DaVinciResolveScript API."""

    def __init__(self) -> None:
        """Initialize Resolve scripting instance."""
        try:
            self._resolve = _get_resolve()
        except Exception as exc:
            raise RuntimeError(
                "DaVinci Resolve scripting API is not available. Start Resolve and enable scripting."
            ) from exc

    def is_available(self) -> bool:
        """Return True if Resolve scripting API is still reachable."""
        try:
            manager = self._resolve.GetProjectManager()
            return manager is not None
        except Exception:
            return False

    def get_current_project(self) -> Any:
        """Return the current Resolve project."""
        manager = self._resolve.GetProjectManager()
        project = manager.GetCurrentProject()
        if project is None:
            project = manager.CreateProject("VideoForge")
        return project

    def get_current_timeline(self) -> Any:
        """Return the active timeline."""
        project = self.get_current_project()
        timeline = project.GetCurrentTimeline()
        if timeline is None:
            raise RuntimeError("No active timeline. Create or open a timeline first.")
        return timeline

    def get_selected_clips(self) -> List[Any]:
        """Return the currently selected timeline clips."""
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
        selected = self.get_selected_clips()
        if selected:
            return selected[0]
        return self.get_clip_at_playhead()

    def create_subtitles_from_audio(self, language: str = "auto") -> bool:
        """Run Resolve Studio auto-captioning on the current timeline."""
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
        fps = self.get_timeline_fps()
        start, end = self._get_item_range(item)
        if start is None or end is None or fps <= 0:
            return None
        return float(start) / float(fps), float(end) / float(fps)

    def get_clip_source_range_seconds(self, item: Any) -> tuple[float, float] | None:
        """Return source in/out range for a timeline item in seconds."""
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
    ) -> None:
        """Insert a media pool clip at a timeline position (frames)."""
        media_pool = self.get_media_pool()
        media_pool.InsertClips([clip], track_type, track_index, timeline_position)

    def get_timeline_fps(self) -> float:
        """Return timeline FPS or 24 if missing."""
        project = self.get_current_project()
        try:
            fps = float(project.GetSetting("timelineFrameRate"))
        except Exception:
            fps = 24.0
        return fps

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
