import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from VideoForge.adapters import embedding_adapter
from VideoForge.broll.db import LibraryDB
from VideoForge.broll.matcher import BrollMatcher
from VideoForge.core.segment_store import SegmentStore
from VideoForge.core.silence_processor import process_silence
from VideoForge.core.timeline_builder import TimelineBuilder
from VideoForge.core.srt_parser import parse_srt
from VideoForge.core.transcriber import align_sentences_to_srt, transcribe_segments
from VideoForge.integrations.resolve_api import ResolveAPI
from VideoForge.plugin.project_manager import ProjectManager
from VideoForge.errors import AnalysisError, MatchError, TimelineError


class ResolveBridge:
    """Bridge for running VideoForge core logic inside Resolve."""

    def __init__(self, settings: Dict) -> None:
        self.resolve_api = ResolveAPI()
        self.settings = settings
        self.project_manager = ProjectManager(self.resolve_api)
        self._ensure_logging()
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _ensure_logging() -> None:
        root_logger = logging.getLogger()
        log_path = os.environ.get("VIDEOFORGE_LOG_PATH")
        if not log_path:
            appdata = os.environ.get("APPDATA", ".")
            log_path = str(
                Path(appdata)
                / "Blackmagic Design"
                / "DaVinci Resolve"
                / "Support"
                / "VideoForge.log"
            )
        try:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            log_path = "VideoForge.log"

        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        logging.raiseExceptions = False
        file_handler = logging.FileHandler(
            log_path, encoding="utf-8", delay=True, mode="w"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        for noisy in ("urllib3", "httpx", "httpcore"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        root_logger.info("VideoForge logging initialized: %s", log_path)

    def analyze_selected_clip(
        self,
        whisper_model: str = "large-v3",
        whisper_task: str = "translate",
        whisper_language: str | None = None,
        align_srt: str | None = None,
    ) -> Dict[str, str]:
        """Analyze the selected timeline clip and store segments/sentences."""
        main_clip = self.resolve_api.get_primary_clip()
        if not main_clip:
            raise ValueError("No clip selected in the current timeline.")

        video_path = self._get_clip_path(main_clip)
        if not video_path:
            raise ValueError("Selected clip has no file path.")
        return self.analyze_from_path(
            video_path=video_path,
            whisper_model=whisper_model,
            whisper_task=whisper_task,
            whisper_language=whisper_language,
            align_srt=align_srt,
        )

    def analyze_from_path(
        self,
        video_path: str,
        whisper_model: str = "large-v3",
        whisper_task: str = "translate",
        whisper_language: str | None = None,
        align_srt: str | None = None,
        engine: str = "whisper",
        clip_range: tuple[float, float] | None = None,
    ) -> Dict[str, str]:
        """Analyze from already-extracted video_path (thread-safe version)."""
        project_id = self._generate_project_id(video_path)
        project_db = self._get_project_db_path(project_id)
        self.logger.info("Analyze start: %s", video_path)

        store = SegmentStore(project_db)
        store.clear_analysis_data()
        store.save_artifact("project_id", project_id)
        store.save_artifact("whisper_model", whisper_model)
        store.save_artifact("render_mode", None)
        store.save_artifact("render_segments", None)
        store.save_artifact("render_overlap_frames", None)
        store.save_artifact("render_source_range", None)

        try:
            self.logger.info("=== [BRIDGE] analyze_from_path called ===")
            self.logger.info("[BRIDGE] video_path: %s", video_path)
            self.logger.info(
                "[BRIDGE] whisper_model: %s, task: %s, language: %s",
                whisper_model,
                whisper_task,
                whisper_language or "auto",
            )
            if engine == "resolve_ai":
                raise RuntimeError("Resolve AI engine must run on the UI thread.")

            self.logger.info("Transcribing full audio (no silence removal).")
            if clip_range:
                full_segment = [{"id": 0, "t0": float(clip_range[0]), "t1": float(clip_range[1])}]
            else:
                full_segment = [{"id": 0, "t0": 0.0, "t1": 10**9}]
            self.logger.info("[BRIDGE] Calling transcribe_segments (Whisper will load now)")
            self.logger.info("[BRIDGE] NOTE: First run may take 1-2 minutes to download model")
            sentences = transcribe_segments(
                video_path,
                full_segment,
                whisper_model,
                task=whisper_task,
                language=whisper_language,
                clip_range=clip_range,
            )
            self.logger.info("[BRIDGE] transcribe_segments returned %d sentences", len(sentences))
            self.logger.info("[BRIDGE] Waiting for any background cleanup... (2 seconds)")
            import time
            time.sleep(2)
            self.logger.info("[BRIDGE] Processing post-transcription steps...")
            if self.settings.get("silence", {}).get("enable_removal", False):
                self.logger.info("Applying silence removal to sentences.")
                segments = process_silence(
                    video_path=video_path,
                    threshold_db=self.settings["silence"]["threshold_db"],
                    min_keep=self.settings["silence"]["min_keep_duration"],
                    buffer_before=self.settings["silence"]["buffer_before"],
                    buffer_after=self.settings["silence"]["buffer_after"],
                )
                if clip_range:
                    segments = [
                        seg
                        for seg in segments
                        if float(seg["t1"]) >= clip_range[0] and float(seg["t0"]) <= clip_range[1]
                    ]
                self._assign_segments(sentences, segments)
            else:
                last_t1 = sentences[-1]["t1"] if sentences else 0.0
                segments = [
                    {
                        "id": 0,
                        "t0": float(clip_range[0]) if clip_range else 0.0,
                        "t1": float(clip_range[1]) if clip_range else float(last_t1),
                        "type": "keep",
                        "metadata": {"auto_generated": True},
                    }
                ]
            self.logger.info("[BRIDGE] Segments created: %d", len(segments))
            if align_srt:
                self.logger.info("Aligning sentences to SRT: %s", align_srt)
                srt_entries = parse_srt(align_srt)
                sentences = align_sentences_to_srt(sentences, srt_entries)
                self.logger.info("[BRIDGE] SRT alignment complete")
            self.logger.info("[BRIDGE] Saving to database...")
            store.save_project_data(segments=segments, sentences=sentences)
            self.logger.info("[BRIDGE] Database save complete")
        except Exception as exc:
            self.logger.error("Analysis failed: %s", exc, exc_info=True)
            raise AnalysisError(
                "Analysis failed",
                details={"video_path": video_path, "engine": engine, "error": str(exc)},
            ) from exc

        warning = ""
        if any(seg.get("metadata", {}).get("warning") for seg in segments):
            warning = "Silence detection failed. Using full duration."

        self.logger.info("[BRIDGE] Analyze complete. Segments: %s, Sentences: %s", len(segments), len(sentences))
        return {"project_db": project_db, "warning": warning}

    def analyze_resolve_ai(
        self,
        video_path: str,
        language: str = "auto",
        clip_range: tuple[float, float] | None = None,
        source_range: tuple[float, float] | None = None,
    ) -> Dict[str, str]:
        """Analyze using Resolve Studio auto-captioning (must run on UI thread)."""
        project_id = self._generate_project_id(video_path)
        project_db = self._get_project_db_path(project_id)
        self.logger.info("Analyze(Resolve AI) start: %s", video_path)
        self.logger.info("Analyze(Resolve AI) project_db: %s", project_db)

        try:
            ok = self.resolve_api.create_subtitles_from_audio(language=language)
            if not ok:
                raise RuntimeError("Resolve CreateSubtitlesFromAudio returned False")

            self.logger.info("[BRIDGE] Resolve AI subtitle creation complete")
            if clip_range:
                self.logger.info(
                    "[BRIDGE] Resolve AI clip range filter: %.2fs - %.2fs",
                    clip_range[0],
                    clip_range[1],
                )
            sentences = self.resolve_api.export_subtitles_as_sentences(clip_range=clip_range)
            self.logger.info("[BRIDGE] Resolve AI subtitles extracted: %d", len(sentences))
            if clip_range and source_range:
                try:
                    offset = float(source_range[0]) - float(clip_range[0])
                except Exception:
                    offset = 0.0
                if abs(offset) > 1e-3:
                    self.logger.info(
                        "[BRIDGE] Resolve AI time offset: %.3fs (timeline->source)",
                        offset,
                    )
                    for sentence in sentences:
                        sentence["t0"] = float(sentence.get("t0", 0.0)) + offset
                        sentence["t1"] = float(sentence.get("t1", 0.0)) + offset
                        metadata = sentence.get("metadata") or {}
                        metadata["timeline_offset"] = float(clip_range[0])
                        metadata["source_offset"] = float(source_range[0])
                        sentence["metadata"] = metadata

            if self.settings.get("silence", {}).get("enable_removal", False):
                self.logger.info("[BRIDGE] Applying silence removal after Resolve AI")
                segments = process_silence(
                    video_path=video_path,
                    threshold_db=self.settings["silence"]["threshold_db"],
                    min_keep=self.settings["silence"]["min_keep_duration"],
                    buffer_before=self.settings["silence"]["buffer_before"],
                    buffer_after=self.settings["silence"]["buffer_after"],
                )
                self._assign_segments(sentences, segments)
            else:
                segments = [
                    {
                        "id": 0,
                        "t0": 0.0,
                        "t1": float(sentences[-1]["t1"]) if sentences else 0.0,
                        "type": "keep",
                        "metadata": {"auto_generated": True, "source": "resolve_ai"},
                    }
                ]
            self.logger.info("[BRIDGE] Resolve AI segments created: %d", len(segments))

            store = SegmentStore(project_db)
            store.clear_analysis_data()
            store.save_artifact("project_id", project_id)
            store.save_artifact("whisper_model", "resolve_ai")
            store.save_artifact("render_mode", None)
            store.save_artifact("render_segments", None)
            store.save_artifact("render_overlap_frames", None)
            store.save_artifact("render_source_range", None)
            self.logger.info("[BRIDGE] Resolve AI saving to database...")
            store.save_project_data(segments=segments, sentences=sentences)
            self.logger.info("[BRIDGE] Resolve AI database save complete")
            return {"project_db": project_db, "warning": ""}
        except Exception as exc:
            self.logger.error("Resolve AI analysis failed: %s", exc, exc_info=True)
            raise AnalysisError(
                "Resolve AI analysis failed",
                details={"video_path": video_path, "engine": "resolve_ai", "error": str(exc)},
            ) from exc

    def match_broll(
        self,
        project_db_path: str,
        library_db_path: str,
        sentence_ids: Optional[Iterable[int]] = None,
    ) -> Dict[str, str | int]:
        """Match B-roll clips against stored sentences."""
        store = SegmentStore(project_db_path)
        store.clear_matches()
        sentences = store.get_sentences()
        if not sentences:
            raise ValueError("No sentences found. Run analysis first.")
        if sentence_ids is not None:
            id_set = {int(sid) for sid in sentence_ids if sid is not None}
            total = len(sentences)
            sentences = [s for s in sentences if int(s.get("id") or 0) in id_set]
            self.logger.info("Match scope: %d/%d sentences", len(sentences), total)
            if not sentences:
                self.logger.info("Match scope empty; skipping match.")
                return {"count": 0, "warning": ""}

        schema_path = self._resolve_schema_path()
        library_db = LibraryDB(library_db_path, schema_path=str(schema_path))
        matcher = BrollMatcher(library_db, self.settings)
        try:
            matches = matcher.match(sentences)
            store.save_project_data(matches=matches)
        except Exception as exc:
            self.logger.error("Match failed: %s", exc, exc_info=True)
            raise MatchError(
                "B-roll matching failed",
                details={"project_db": project_db_path, "library_db": library_db_path, "error": str(exc)},
            ) from exc
        warning = ""
        if embedding_adapter.consume_fallback_used():
            warning = "CLIP model unavailable. Using deterministic embeddings."
        self.logger.info("Matches: %s", len(matches))
        return {"count": len(matches), "warning": warning}

    def apply_to_timeline(self, project_db_path: str, main_video_path: str) -> None:
        """Apply VideoForge timeline data directly to Resolve timeline."""
        store = SegmentStore(project_db_path)
        builder = TimelineBuilder(store, self.settings)
        timeline_data = builder.build(main_video_path)
        self.logger.info("Apply to timeline: %s", main_video_path)

        try:
            timeline = self.resolve_api.get_current_timeline()
            fps = self.resolve_api.get_timeline_fps()
            base_offset_frames, base_video_index, skip_main = self._resolve_apply_anchor(
                timeline, main_video_path
            )

            for track in timeline_data["tracks"]:
                track_type = "video"
                if skip_main and track.get("type") == "video_main":
                    self.logger.info("Apply: skipping main track (existing clip detected)")
                    continue
                track_index = int(track["index"])
                if base_video_index:
                    track_index = int(base_video_index) + (track_index - 1)
                self.resolve_api.ensure_track(track_type, track_index)
                for clip_data in track["clips"]:
                    media_item = self.resolve_api.import_media_if_needed(clip_data["file"])
                    if media_item is None:
                        continue
                    timeline_pos = self._time_to_frames(clip_data["timeline_start"], fps)
                    if base_offset_frames:
                        timeline_pos += int(base_offset_frames)
                    clip_duration = float(clip_data.get("duration") or 0.0)
                    target_frames = (
                        max(1, int(round(clip_duration * fps)))
                        if clip_duration > 0 and fps
                        else None
                    )
                    target_track_index = track_index
                    if track["type"] == "video_broll" and target_frames:
                        end_frame = timeline_pos + target_frames - 1
                        target_track_index = self._resolve_non_overlapping_track(
                            timeline,
                            track_type,
                            track_index,
                            timeline_pos,
                            end_frame,
                        )
                        if target_track_index != track_index:
                            self.logger.info(
                                "Apply: overlap detected on V%s; stacking on V%s",
                                track_index,
                                target_track_index,
                            )
                    inserted = None
                    if track["type"] == "video_broll":
                        target_duration = float(clip_data.get("duration") or 0.0)
                        if target_duration > 0 and fps:
                            media_frames = self.resolve_api._get_media_item_duration_frames(
                                media_item, fps
                            )
                            if media_frames:
                                target_frames = max(1, int(target_duration * fps))
                                source_end = min(media_frames - 1, target_frames - 1)
                                inserted = self.resolve_api.insert_clip_at_position_with_range(
                                    track_type,
                                    target_track_index,
                                    media_item,
                                    timeline_pos,
                                    0,
                                    source_end,
                                )
                    if inserted is None:
                        self.resolve_api.insert_clip_at_position(
                            track_type, target_track_index, media_item, timeline_pos
                        )

                    if track["type"] == "video_broll" and not clip_data.get("audio_enabled", False):
                        items = timeline.GetItemListInTrack(track_type, target_track_index)
                        if items:
                            last_item = items[-1]
                            try:
                                last_item.SetProperty("Audio Enabled", False)
                            except Exception:
                                pass
        except Exception as exc:
            self.logger.error("Timeline apply failed: %s", exc, exc_info=True)
            raise TimelineError(
                "Apply to timeline failed",
                details={"video_path": main_video_path, "error": str(exc)},
            ) from exc

    def _resolve_apply_anchor(
        self, timeline: Any, main_video_path: str
    ) -> tuple[int, int, bool]:
        base_offset_frames = 0
        base_video_index = 1
        skip_main = False
        target_norm = os.path.normcase(str(main_video_path))

        primary = self.resolve_api.get_primary_clip()
        primary_path = self._get_clip_path(primary) if primary else None
        if primary_path and os.path.normcase(str(primary_path)) == target_norm:
            start_frame, _end_frame = self.resolve_api._get_item_range(primary)
            if start_frame is not None:
                base_offset_frames = int(start_frame)
            base_video_index = int(self.resolve_api._safe_call(primary, "GetTrackIndex") or 1)
            skip_main = True
            self.logger.info(
                "Apply anchor: selected clip at frame=%s track=%s",
                base_offset_frames,
                base_video_index,
            )
            return base_offset_frames, base_video_index, skip_main

        track_count = int(timeline.GetTrackCount("video") or 0)
        for track_index in range(1, track_count + 1):
            items = timeline.GetItemListInTrack("video", track_index) or []
            for item in items:
                path = self._get_clip_path(item)
                if not path:
                    continue
                if os.path.normcase(str(path)) != target_norm:
                    continue
                start_frame, _end_frame = self.resolve_api._get_item_range(item)
                if start_frame is not None:
                    base_offset_frames = int(start_frame)
                base_video_index = int(self.resolve_api._safe_call(item, "GetTrackIndex") or track_index)
                skip_main = True
                self.logger.info(
                    "Apply anchor: matched clip at frame=%s track=%s",
                    base_offset_frames,
                    base_video_index,
                )
                return base_offset_frames, base_video_index, skip_main

        return base_offset_frames, base_video_index, skip_main

    def _resolve_non_overlapping_track(
        self,
        timeline: Any,
        track_type: str,
        start_index: int,
        start_frame: int,
        end_frame: int,
        max_layers: int = 6,
    ) -> int:
        target_index = int(start_index)
        for _ in range(max_layers):
            if not self._track_has_overlap(
                timeline, track_type, target_index, start_frame, end_frame
            ):
                return target_index
            target_index += 1
            self.resolve_api.ensure_track(track_type, target_index)
        return target_index

    def _track_has_overlap(
        self,
        timeline: Any,
        track_type: str,
        track_index: int,
        start_frame: int,
        end_frame: int,
    ) -> bool:
        items = timeline.GetItemListInTrack(track_type, int(track_index)) or []
        for item in items:
            item_start, item_end = self.resolve_api._get_item_range(item)
            if item_start is None or item_end is None:
                continue
            if item_start <= end_frame and item_end >= start_frame:
                return True
        return False

    def save_library_path(self, library_path: str) -> None:
        """Persist library path for the current Resolve project."""
        if not library_path:
            return
        self.project_manager.save_library_path(library_path)

    def get_saved_library_path(self) -> str | None:
        """Return saved library path if available."""
        return self.project_manager.load_library_path()

    def _generate_project_id(self, video_path: str) -> str:
        """Generate a stable project id from path, size, and mtime."""
        stat = os.stat(video_path)
        payload = f"{video_path}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()

    def _get_project_db_path(self, project_id: str) -> str:
        """Return the project DB path under data/projects."""
        base = Path(__file__).resolve().parents[1] / "data" / "projects"
        base.mkdir(parents=True, exist_ok=True)
        return str(base / f"{project_id}.db")

    @staticmethod
    def _time_to_frames(seconds: float, fps: float) -> int:
        return int(round(seconds * fps))

    @staticmethod
    def _get_clip_path(clip: Any) -> str | None:
        getter = getattr(clip, "GetClipProperty", None)
        if callable(getter):
            try:
                return getter("File Path")
            except Exception:
                pass
        media_item = getattr(clip, "GetMediaPoolItem", None)
        if callable(media_item):
            try:
                item = media_item()
            except Exception:
                item = None
            getter = getattr(item, "GetClipProperty", None) if item else None
            if callable(getter):
                try:
                    return getter("File Path")
                except Exception:
                    pass
        return None

    @staticmethod
    def _assign_segments(sentences: List[Dict], segments: List[Dict]) -> None:
        for sentence in sentences:
            seg_id = None
            t0 = float(sentence.get("t0", 0.0))
            for seg in segments:
                if float(seg["t0"]) <= t0 <= float(seg["t1"]):
                    seg_id = int(seg.get("id")) if seg.get("id") is not None else None
                    break
            sentence["segment_id"] = seg_id

    @staticmethod
    def _resolve_schema_path() -> Path:
        root = Path(__file__).resolve().parents[1]
        schema = root / "config" / "schema.sql"
        if not schema.exists():
            raise FileNotFoundError("schema.sql not found. Check VideoForge installation.")
        return schema
