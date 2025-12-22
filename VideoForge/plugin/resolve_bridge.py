import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, List

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
        file_handler = logging.FileHandler(log_path, encoding="utf-8", delay=True)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.info("VideoForge logging initialized: %s", log_path)

    def analyze_selected_clip(
        self,
        whisper_model: str = "large-v3",
        whisper_task: str = "translate",
        whisper_language: str | None = None,
        align_srt: str | None = None,
    ) -> Dict[str, str]:
        """Analyze the selected timeline clip and store segments/sentences."""
        clips = self.resolve_api.get_selected_clips()
        if not clips:
            raise ValueError("No clip selected in the current timeline.")

        main_clip = clips[0]
        video_path = main_clip.GetClipProperty("File Path")
        if not video_path:
            raise ValueError("Selected clip has no file path.")

        project_id = self._generate_project_id(video_path)
        project_db = self._get_project_db_path(project_id)
        self.logger.info("Analyze start: %s", video_path)

        store = SegmentStore(project_db)
        store.save_artifact("project_id", project_id)
        store.save_artifact("whisper_model", whisper_model)

        try:
            self.logger.info("=== [BRIDGE] analyze_selected_clip called ===")
            self.logger.info("[BRIDGE] video_path: %s", video_path)
            self.logger.info(
                "[BRIDGE] whisper_model: %s, task: %s, language: %s",
                whisper_model,
                whisper_task,
                whisper_language or "auto",
            )
            self.logger.info("Transcribing full audio (no silence removal).")
            full_segment = [{"id": 0, "t0": 0.0, "t1": 10**9}]
            self.logger.info("[BRIDGE] Calling transcribe_segments (Whisper will load now)")
            self.logger.info("[BRIDGE] NOTE: First run may take 1-2 minutes to download model")
            sentences = transcribe_segments(
                video_path,
                full_segment,
                whisper_model,
                task=whisper_task,
                language=whisper_language,
            )
            self.logger.info("[BRIDGE] transcribe_segments returned %d sentences", len(sentences))
            if self.settings.get("silence", {}).get("enable_removal", False):
                self.logger.info("Applying silence removal to sentences.")
                segments = process_silence(
                    video_path=video_path,
                    threshold_db=self.settings["silence"]["threshold_db"],
                    min_keep=self.settings["silence"]["min_keep_duration"],
                    buffer_before=self.settings["silence"]["buffer_before"],
                    buffer_after=self.settings["silence"]["buffer_after"],
                )
                self._assign_segments(sentences, segments)
            else:
                last_t1 = sentences[-1]["t1"] if sentences else 0.0
                segments = [
                    {
                        "id": 0,
                        "t0": 0.0,
                        "t1": float(last_t1),
                        "type": "keep",
                        "metadata": {"auto_generated": True},
                    }
                ]
            if align_srt:
                self.logger.info("Aligning sentences to SRT: %s", align_srt)
                srt_entries = parse_srt(align_srt)
                sentences = align_sentences_to_srt(sentences, srt_entries)
            store.save_project_data(segments=segments, sentences=sentences)
        except Exception as exc:
            self.logger.error("Analysis failed: %s", exc, exc_info=True)
            raise

        warning = ""
        if any(seg.get("metadata", {}).get("warning") for seg in segments):
            warning = "Silence detection failed. Using full duration."

        self.logger.info("Silence segments: %s", len(segments))
        self.logger.info("Sentences: %s", len(sentences))
        return {"project_db": project_db, "warning": warning}

    def match_broll(self, project_db_path: str, library_db_path: str) -> Dict[str, str | int]:
        """Match B-roll clips against stored sentences."""
        store = SegmentStore(project_db_path)
        sentences = store.get_sentences()
        if not sentences:
            raise ValueError("No sentences found. Run analysis first.")

        schema_path = self._resolve_schema_path()
        library_db = LibraryDB(library_db_path, schema_path=str(schema_path))
        matcher = BrollMatcher(library_db, self.settings)
        try:
            matches = matcher.match(sentences)
            store.save_project_data(matches=matches)
        except Exception as exc:
            self.logger.error("Match failed: %s", exc, exc_info=True)
            raise
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

        timeline = self.resolve_api.get_current_timeline()
        fps = self.resolve_api.get_timeline_fps()

        for track in timeline_data["tracks"]:
            track_type = "video"
            track_index = int(track["index"])
            self.resolve_api.ensure_track(track_type, track_index)
            for clip_data in track["clips"]:
                media_item = self.resolve_api.import_media_if_needed(clip_data["file"])
                if media_item is None:
                    continue
                timeline_pos = self._time_to_frames(clip_data["timeline_start"], fps)
                self.resolve_api.insert_clip_at_position(track_type, track_index, media_item, timeline_pos)

                if track["type"] == "video_broll" and not clip_data.get("audio_enabled", False):
                    items = timeline.GetItemListInTrack(track_type, track_index)
                    if items:
                        last_item = items[-1]
                        try:
                            last_item.SetProperty("Audio Enabled", False)
                        except Exception:
                            pass

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
