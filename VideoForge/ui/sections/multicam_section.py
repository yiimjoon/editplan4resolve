"""Multicam auto-cut UI section."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from VideoForge.config.config_manager import Config
from VideoForge.core.segment_store import SegmentStore
from VideoForge.multicam.angle_scorer import AngleScorer
from VideoForge.multicam.angle_selector import AngleSelector
from VideoForge.multicam.boundary_detector import BoundaryDetector
from VideoForge.multicam.llm_tagger import LLMTagger
from VideoForge.multicam.plan_generator import MulticamPlanGenerator
from VideoForge.ui.qt_compat import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSlider,
    QTextBrowser,
    QThread,
    QVBoxLayout,
    QWidget,
    Qt,
    Signal,
)

logger = logging.getLogger("VideoForge.ui.multicam")


class MulticamWorker(QThread):
    progress_updated = Signal(int)
    plan_ready = Signal(object)
    error_occurred = Signal(str)

    def __init__(
        self,
        timeline_info: Dict,
        track_items: List[Dict],
        project_db: str,
        mode: str,
        use_face: bool,
        use_llm: bool,
    ) -> None:
        super().__init__()
        self.timeline_info = timeline_info
        self.track_items = track_items
        self.project_db = project_db
        self.mode = mode
        self.use_face = use_face
        self.use_llm = use_llm

    def run(self) -> None:
        try:
            self.progress_updated.emit(10)
            sentences = self._load_sentences()
            if not sentences:
                raise RuntimeError("No transcript found. Run Analyze first.")

            self.progress_updated.emit(20)
            detector = BoundaryDetector()
            cut_points = detector.extract_from_transcript(sentences, mode=self.mode)
            segments = self._build_segments(cut_points, sentences)
            if not segments:
                raise RuntimeError("No segments available for multicam scoring.")

            self.progress_updated.emit(40)
            scored_segments = self._score_segments(segments)

            segment_tags = None
            if self.use_llm:
                self.progress_updated.emit(70)
                tagger = LLMTagger()
                segment_tags = tagger.tag_segments(scored_segments)

            self.progress_updated.emit(80)
            selector = AngleSelector()
            selections = selector.select_angles(scored_segments, segment_tags)
            if not selections:
                raise RuntimeError("Angle selection produced no cuts.")

            self.progress_updated.emit(90)
            generator = MulticamPlanGenerator()
            source_clips = self._get_source_clip_names()
            plan = generator.generate_plan(
                selections, self.timeline_info, source_clips
            )

            self.progress_updated.emit(100)
            self.plan_ready.emit(plan)
        except Exception as exc:
            logger.error("Multicam worker failed: %s", exc, exc_info=True)
            self.error_occurred.emit(str(exc))

    def _load_sentences(self) -> List[Dict]:
        store = SegmentStore(self.project_db)
        return [s for s in store.get_sentences() if str(s.get("text", "")).strip()]

    @staticmethod
    def _build_segments(cut_points: List[float], sentences: List[Dict]) -> List[Dict]:
        if not cut_points or len(cut_points) < 2:
            return []
        segments: List[Dict] = []
        for start, end in zip(cut_points, cut_points[1:]):
            text_parts = [
                str(s.get("text", "")).strip()
                for s in sentences
                if float(s.get("t0", 0.0)) < end and float(s.get("t1", 0.0)) > start
            ]
            text = " ".join([t for t in text_parts if t])
            segments.append(
                {
                    "start_sec": float(start),
                    "end_sec": float(end),
                    "text": text,
                    "scores": {},
                }
            )
        return segments

    def _score_segments(self, segments: List[Dict]) -> List[Dict]:
        scorer = AngleScorer()
        video_tracks = int(self.timeline_info.get("video_tracks", 0))
        total = max(1, len(segments))
        for idx, seg in enumerate(segments):
            progress = 30 + int(40 * (idx / total))
            self.progress_updated.emit(progress)
            start_sec = float(seg.get("start_sec", 0.0))
            end_sec = float(seg.get("end_sec", 0.0))
            mid_sec = (start_sec + end_sec) / 2.0
            scores: Dict[int, Dict[str, float]] = {}
            for angle_idx in range(video_tracks):
                track_index = angle_idx + 1
                item = self._find_track_item(track_index, mid_sec)
                metrics = {
                    "sharpness": 0.0,
                    "motion": 0.0,
                    "stability": 0.0,
                    "face_score": 0.0,
                }
                if item and item.get("media_path"):
                    metrics = self._score_item(item, mid_sec, scorer)
                if not self.use_face:
                    metrics["face_score"] = 0.0
                scores[angle_idx] = metrics
            seg["scores"] = scores
        return segments

    def _score_item(self, item: Dict, mid_sec: float, scorer: AngleScorer) -> Dict[str, float]:
        timeline_start = float(item.get("timeline_start", 0.0))
        source_start = float(item.get("source_start", timeline_start))
        source_end = float(item.get("source_end", source_start))
        source_time = source_start + max(0.0, mid_sec - timeline_start)
        if source_end > source_start:
            source_time = min(source_time, source_end)
        media_path = Path(item["media_path"])
        try:
            from VideoForge.adapters.opencv_subprocess import should_use_subprocess

            if should_use_subprocess():
                try:
                    return scorer.score_video_frame(media_path, source_time)
                except Exception:
                    return {
                        "sharpness": 0.0,
                        "motion": 0.0,
                        "stability": 0.0,
                        "face_score": 0.0,
                    }
        except Exception:
            pass

        frame = self._extract_frame_at(media_path, source_time)
        if frame is None:
            return {
                "sharpness": 0.0,
                "motion": 0.0,
                "stability": 0.0,
                "face_score": 0.0,
            }
        try:
            return scorer.score_frame(frame)
        except Exception:
            return {
                "sharpness": 0.0,
                "motion": 0.0,
                "stability": 0.0,
                "face_score": 0.0,
            }

    def _find_track_item(self, track_index: int, time_sec: float) -> Optional[Dict]:
        for item in self.track_items:
            if int(item.get("track_index", 0)) != int(track_index):
                continue
            start = float(item.get("timeline_start", 0.0))
            end = float(item.get("timeline_end", 0.0))
            if start <= time_sec <= end:
                return item
        return None

    @staticmethod
    def _extract_frame_at(media_path: Path, timestamp_sec: float):
        try:
            from VideoForge.adapters.video_analyzer import VideoAnalyzer

            return VideoAnalyzer.extract_frame_at_time(media_path, timestamp_sec)
        except Exception:
            return None

    def _get_source_clip_names(self) -> List[str]:
        video_tracks = int(self.timeline_info.get("video_tracks", 0))
        clip_names: List[str] = []
        for track_index in range(1, video_tracks + 1):
            clip_name = None
            for item in self.track_items:
                if int(item.get("track_index", 0)) != track_index:
                    continue
                media_path = item.get("media_path")
                if media_path:
                    clip_name = Path(media_path).name
                    break
                if item.get("clip_name"):
                    clip_name = str(item["clip_name"])
                    break
            if not clip_name:
                clip_name = f"Angle {track_index}"
            clip_names.append(clip_name)
        return clip_names


class MulticamSection(QWidget):
    """Multicam auto-cut UI."""

    plan_generated = Signal(object)

    def __init__(self, resolve_api) -> None:
        super().__init__()
        self.resolve_api = resolve_api
        self.worker: Optional[MulticamWorker] = None
        self.pending_plan = None
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        settings_group = QGroupBox("Multicam Cut Settings")
        settings_layout = QFormLayout()

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Hybrid (Recommended)", "Sentence Only", "Fixed Interval"])
        mode_map = {
            "hybrid": "Hybrid (Recommended)",
            "sentence": "Sentence Only",
            "fixed": "Fixed Interval",
        }
        current_mode = str(Config.get("multicam_boundary_mode", "hybrid")).strip().lower()
        self.mode_combo.setCurrentText(mode_map.get(current_mode, "Hybrid (Recommended)"))
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        settings_layout.addRow("Cut Boundary Mode:", self.mode_combo)

        self.max_segment_slider = self._make_slider(5, 30, int(Config.get("multicam_max_segment_sec", 10)))
        self.max_segment_label = QLabel(f"{self.max_segment_slider.value()}s")
        self.max_segment_slider.valueChanged.connect(
            lambda v: Config.set("multicam_max_segment_sec", float(v))
        )
        self.max_segment_slider.valueChanged.connect(
            lambda v: self.max_segment_label.setText(f"{v}s")
        )
        settings_layout.addRow("Max Segment Length:", self.max_segment_slider)
        settings_layout.addRow("", self.max_segment_label)

        self.min_hold_slider = self._make_slider(10, 50, int(float(Config.get("multicam_min_hold_sec", 2.0)) * 10))
        self.min_hold_label = QLabel(f"{self.min_hold_slider.value() / 10.0}s")
        self.min_hold_slider.valueChanged.connect(
            lambda v: Config.set("multicam_min_hold_sec", v / 10.0)
        )
        self.min_hold_slider.valueChanged.connect(
            lambda v: self.min_hold_label.setText(f"{v / 10.0}s")
        )
        settings_layout.addRow("Min Hold Time:", self.min_hold_slider)
        settings_layout.addRow("", self.min_hold_label)

        self.face_detection_check = QCheckBox("Enable Face Detection")
        face_detector = str(Config.get("multicam_face_detector", "opencv_dnn")).strip().lower()
        self.face_detection_check.setChecked(face_detector != "disabled")
        settings_layout.addRow("", self.face_detection_check)

        self.llm_tagging_check = QCheckBox("Enable LLM Tagging (Experimental)")
        llm_enabled = str(Config.get("llm_provider", "disabled")) != "disabled"
        self.llm_tagging_check.setEnabled(llm_enabled)
        self.llm_tagging_check.setChecked(False)
        settings_layout.addRow("", self.llm_tagging_check)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        self.generate_btn = QPushButton("Generate Multicam Cuts")
        self.generate_btn.clicked.connect(self._on_generate_cuts)
        layout.addWidget(self.generate_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        preview_group = QGroupBox("Cut Preview")
        preview_layout = QVBoxLayout()

        self.preview_text = QTextBrowser()
        self.preview_text.setMaximumHeight(200)
        preview_layout.addWidget(self.preview_text)

        approve_layout = QHBoxLayout()
        self.approve_btn = QPushButton("Approve & Apply")
        self.approve_btn.clicked.connect(self._on_approve)
        self.approve_btn.setEnabled(False)
        self.reject_btn = QPushButton("Reject")
        self.reject_btn.clicked.connect(self._on_reject)
        self.reject_btn.setEnabled(False)
        approve_layout.addWidget(self.approve_btn)
        approve_layout.addWidget(self.reject_btn)
        preview_layout.addLayout(approve_layout)

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        layout.addStretch()
        self.setLayout(layout)

        self._on_mode_changed(self.mode_combo.currentText())

    def _on_mode_changed(self, text: str) -> None:
        mode_map = {
            "Hybrid (Recommended)": "hybrid",
            "Sentence Only": "sentence",
            "Fixed Interval": "fixed",
        }
        Config.set("multicam_boundary_mode", mode_map.get(text, "hybrid"))

    def _on_generate_cuts(self) -> None:
        try:
            timeline_info, track_items = self._build_timeline_snapshot()
        except Exception as exc:
            self.preview_text.setHtml(f"<b>Error:</b> {exc}")
            return

        if timeline_info.get("video_tracks", 0) < 2:
            self.preview_text.setHtml(
                "<b>Error:</b> Need at least 2 video tracks for multicam"
            )
            return

        project_db = self._resolve_project_db()
        if not project_db:
            self.preview_text.setHtml(
                "<b>Error:</b> No transcript found. Run Analyze first."
            )
            return

        self.generate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        mode = str(Config.get("multicam_boundary_mode", "hybrid"))
        self.worker = MulticamWorker(
            timeline_info=timeline_info,
            track_items=track_items,
            project_db=project_db,
            mode=mode,
            use_face=self.face_detection_check.isChecked(),
            use_llm=self.llm_tagging_check.isChecked(),
        )
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.plan_ready.connect(self._on_plan_ready)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.start()

    def _on_plan_ready(self, plan) -> None:
        self.pending_plan = plan
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)

        html = f"<h3>{plan.summary}</h3>"
        html += f"<p><b>Risk Level:</b> {plan.risk_level}</p>"
        html += f"<p><b>Actions:</b> {len(plan.actions)}</p>"
        html += "<hr><h4>Cut Points:</h4><ul>"

        for action in plan.actions[:10]:
            params = action.params
            html += (
                f"<li>[{params['start_sec']:.1f}s - {params['end_sec']:.1f}s] "
                f"-> Angle {params['target_angle_track'] + 1} "
                f"({action.reason})</li>"
            )

        if len(plan.actions) > 10:
            html += f"<li>... and {len(plan.actions) - 10} more cuts</li>"

        html += "</ul>"

        if plan.warnings:
            html += "<hr><h4>Warnings:</h4><ul>"
            for warning in plan.warnings:
                html += f"<li>WARN: {warning}</li>"
            html += "</ul>"

        self.preview_text.setHtml(html)
        self.approve_btn.setEnabled(True)
        self.reject_btn.setEnabled(True)

        self.plan_generated.emit(plan)

    def _on_approve(self) -> None:
        if not self.pending_plan:
            return

        self.approve_btn.setEnabled(False)
        self.reject_btn.setEnabled(False)

        try:
            result = self.resolve_api.apply_multicam_cuts(self.pending_plan)
            if isinstance(result, dict) and result.get("status") == "edl_fallback":
                path = result.get("edl_path")
                message = str(result.get("message", "")).replace("\n", "<br>")
                self.preview_text.setHtml(
                    "<h3>EDL Export Complete</h3>"
                    f"<p>{message}</p>"
                    f"<p><b>File:</b> <code>{path}</code></p>"
                )
            else:
                self.preview_text.setHtml(
                    "<b>Success:</b> Multicam cuts applied to timeline"
                )
        except Exception as exc:
            self.preview_text.setHtml(
                f"<b>Error:</b> Failed to apply cuts: {exc}"
            )

        self.pending_plan = None

    def _on_reject(self) -> None:
        self.pending_plan = None
        self.preview_text.setHtml(
            "<i>Plan rejected. Adjust settings and regenerate.</i>"
        )
        self.approve_btn.setEnabled(False)
        self.reject_btn.setEnabled(False)

    def _on_error(self, error_msg: str) -> None:
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.preview_text.setHtml(f"<b>Error:</b> {error_msg}")

    def _resolve_project_db(self) -> Optional[str]:
        try:
            project_key = self.resolve_api.get_project_key()
        except Exception as exc:
            logger.warning("Failed to read project key: %s", exc)
            return None
        project_map = Config.get("project_db_map", {})
        if not isinstance(project_map, dict):
            return None
        payload = project_map.get(project_key, {})
        if not isinstance(payload, dict):
            return None
        project_db = payload.get("project_db")
        if project_db and Path(str(project_db)).exists():
            return str(project_db)
        return None

    def _build_timeline_snapshot(self) -> tuple[Dict, List[Dict]]:
        timeline = self.resolve_api.get_current_timeline()
        name = "Timeline"
        getter = getattr(timeline, "GetName", None)
        if callable(getter):
            try:
                name = str(getter())
            except Exception:
                pass
        fps = float(self.resolve_api.get_timeline_fps())
        timecode = None
        timecode_getter = getattr(timeline, "GetCurrentTimecode", None)
        if callable(timecode_getter):
            try:
                timecode = str(timecode_getter())
            except Exception:
                timecode = None
        video_tracks = int(timeline.GetTrackCount("video") or 0)
        audio_tracks = int(timeline.GetTrackCount("audio") or 0)
        timeline_info = {
            "name": name,
            "fps": fps,
            "timecode": timecode or "",
            "video_tracks": video_tracks,
            "audio_tracks": audio_tracks,
        }

        track_items: List[Dict] = []
        for track_index in range(1, video_tracks + 1):
            items = timeline.GetItemListInTrack("video", track_index) or []
            for item in items:
                timeline_range = self.resolve_api.get_item_range_seconds(item)
                if not timeline_range:
                    continue
                source_range = (
                    self.resolve_api.get_clip_source_range_seconds(item) or timeline_range
                )
                media_path = self.resolve_api.get_item_media_path(item)
                clip_name = None
                name_getter = getattr(item, "GetName", None)
                if callable(name_getter):
                    try:
                        clip_name = str(name_getter())
                    except Exception:
                        clip_name = None
                track_items.append(
                    {
                        "track_index": int(track_index),
                        "timeline_start": float(timeline_range[0]),
                        "timeline_end": float(timeline_range[1]),
                        "source_start": float(source_range[0]),
                        "source_end": float(source_range[1]),
                        "media_path": media_path,
                        "clip_name": clip_name,
                    }
                )

        if not track_items:
            raise RuntimeError("No clips found on video tracks.")

        return timeline_info, track_items

    @staticmethod
    def _make_slider(min_val: int, max_val: int, current: int):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(min_val))
        slider.setMaximum(int(max_val))
        slider.setValue(int(current))
        return slider
