"""Audio beat engine UI section."""

from __future__ import annotations

import logging
import random
import zlib
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from VideoForge.config.config_manager import Config
from VideoForge.integrations.resolve_api import ResolveAPI
from VideoForge.ui.qt_compat import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QSlider,
    QTextBrowser,
    QThread,
    QVBoxLayout,
    QWidget,
    Qt,
    Signal,
)

logger = logging.getLogger("VideoForge.ui.beat")


class BeatWorker(QThread):
    result_ready = Signal(object)
    error_occurred = Signal(str)

    def __init__(
        self,
        audio_path: Path,
        mode: str,
        hop_length: int,
        onset_threshold: float,
        min_bpm: float,
        max_bpm: float,
    ) -> None:
        super().__init__()
        self.audio_path = Path(audio_path)
        self.mode = mode
        self.hop_length = hop_length
        self.onset_threshold = onset_threshold
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm

    def run(self) -> None:
        try:
            from VideoForge.adapters.beat_subprocess import (
                run_detect_beats_with_debug,
                should_use_subprocess,
            )
            from VideoForge.adapters.beat_detector import BeatDetector

            if should_use_subprocess():
                data, error, debug = run_detect_beats_with_debug(
                    audio_path=self.audio_path,
                    mode=self.mode,
                    hop_length=self.hop_length,
                    onset_threshold=self.onset_threshold,
                    min_bpm=self.min_bpm,
                    max_bpm=self.max_bpm,
                )
                if error:
                    raise RuntimeError(error)
                if isinstance(data, dict):
                    data["debug"] = debug
                self.result_ready.emit(data)
                return

            detector = BeatDetector(
                mode=self.mode,
                hop_length=self.hop_length,
                onset_threshold=self.onset_threshold,
                min_bpm=self.min_bpm,
                max_bpm=self.max_bpm,
            )
            data = detector.detect(self.audio_path, mode=self.mode)
            if isinstance(data, dict):
                data["debug"] = detector.last_debug
            self.result_ready.emit(data)
        except Exception as exc:
            logger.error("Beat detection failed: %s", exc, exc_info=True)
            self.error_occurred.emit(str(exc))


class BeatSection(QWidget):
    """Audio Beat Engine UI."""

    def __init__(self, resolve_api: ResolveAPI) -> None:
        super().__init__()
        self.resolve_api = resolve_api
        self.worker: Optional[BeatWorker] = None
        self.beat_data: Optional[Dict[str, Any]] = None
        self.beat_base_frame: Optional[int] = None
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        settings_group = QGroupBox("Audio Beat Engine")
        settings_layout = QFormLayout()

        self.source_combo = QComboBox()
        self.source_combo.addItems(["Timeline Audio", "Reference Audio File"])
        self.source_combo.currentTextChanged.connect(self._on_source_changed)
        settings_layout.addRow("Source:", self.source_combo)

        source_row = QHBoxLayout()
        self.source_path_edit = QLineEdit()
        self.source_path_edit.setPlaceholderText("Select an audio file...")
        self.source_browse_btn = QPushButton("Browse")
        self.source_browse_btn.clicked.connect(self._on_browse_audio)
        source_row.addWidget(self.source_path_edit)
        source_row.addWidget(self.source_browse_btn)
        settings_layout.addRow("", source_row)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(
            [
                "Onset (Fast Cuts)",
                "Beat (Regular Cuts)",
                "Downbeat (Slow Cuts)",
                "Segment (Intro/Chorus)",
            ]
        )
        mode_map = {
            "onset": "Onset (Fast Cuts)",
            "beat": "Beat (Regular Cuts)",
            "downbeat": "Downbeat (Slow Cuts)",
            "segment": "Segment (Intro/Chorus)",
        }
        current_mode = str(Config.get("beat_detector_mode", "onset")).strip().lower()
        self.mode_combo.setCurrentText(mode_map.get(current_mode, "Onset (Fast Cuts)"))
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        settings_layout.addRow("Detection Mode:", self.mode_combo)

        self.onset_threshold_slider = self._make_slider(
            0, 100, int(float(Config.get("beat_detector_onset_threshold", 0.5)) * 100)
        )
        self.onset_threshold_label = QLabel(
            f"{self.onset_threshold_slider.value() / 100.0:.2f}"
        )
        self.onset_threshold_slider.valueChanged.connect(self._on_onset_threshold_changed)
        settings_layout.addRow("Onset Threshold:", self.onset_threshold_slider)
        settings_layout.addRow("", self.onset_threshold_label)

        self.downbeat_weight_slider = self._make_slider(
            0, 100, int(float(Config.get("beat_detector_downbeat_weight", 0.3)) * 100)
        )
        self.downbeat_weight_label = QLabel(
            f"{self.downbeat_weight_slider.value() / 100.0:.2f}"
        )
        self.downbeat_weight_slider.valueChanged.connect(self._on_downbeat_weight_changed)
        settings_layout.addRow("Downbeat Weight:", self.downbeat_weight_slider)
        settings_layout.addRow("", self.downbeat_weight_label)

        self.use_downbeat_markers = QCheckBox("Add Downbeat Markers (Red)")
        self.use_downbeat_markers.setChecked(True)
        settings_layout.addRow("", self.use_downbeat_markers)

        self.broll_mode_combo = QComboBox()
        self.broll_mode_combo.addItems(
            [
                "Simple Random (Selected Clips)",
                "Transcript-Aware (Planned)",
            ]
        )
        current_broll_mode = str(Config.get("beat_broll_mode", "simple_random"))
        if current_broll_mode.strip().lower() == "transcript":
            self.broll_mode_combo.setCurrentText("Transcript-Aware (Planned)")
        else:
            self.broll_mode_combo.setCurrentText("Simple Random (Selected Clips)")
        self.broll_mode_combo.currentTextChanged.connect(self._on_broll_mode_changed)
        settings_layout.addRow("B-roll Mode:", self.broll_mode_combo)

        self.broll_fill_slider = self._make_slider(
            5, 100, int(float(Config.get("beat_broll_fill_ratio", 1.0)) * 100)
        )
        self.broll_fill_label = QLabel(
            f"{self.broll_fill_slider.value() / 100.0:.2f}"
        )
        self.broll_fill_slider.valueChanged.connect(self._on_broll_fill_changed)
        settings_layout.addRow("B-roll Fill %:", self.broll_fill_slider)
        settings_layout.addRow("", self.broll_fill_label)

        self.broll_seed_edit = QLineEdit()
        self.broll_seed_edit.setPlaceholderText("Random each run")
        settings_layout.addRow("B-roll Seed:", self.broll_seed_edit)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        button_row = QHBoxLayout()
        self.detect_btn = QPushButton("Detect Beats")
        self.detect_btn.clicked.connect(self._on_detect_beats)
        self.add_markers_btn = QPushButton("Add Timeline Markers")
        self.add_markers_btn.clicked.connect(self._on_add_markers)
        self.generate_broll_btn = QPushButton("Generate Beat-Based B-roll")
        self.generate_broll_btn.clicked.connect(self._on_generate_broll)
        self.auto_edit_btn = QPushButton("Auto-Edit to Beat")
        self.auto_edit_btn.clicked.connect(self._on_auto_edit)
        button_row.addWidget(self.detect_btn)
        button_row.addWidget(self.add_markers_btn)
        button_row.addWidget(self.generate_broll_btn)
        button_row.addWidget(self.auto_edit_btn)
        layout.addLayout(button_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.results = QTextBrowser()
        self.results.setMaximumHeight(180)
        layout.addWidget(self.results)

        layout.addStretch()
        self.setLayout(layout)

        self._on_source_changed(self.source_combo.currentText())

    def _on_source_changed(self, text: str) -> None:
        is_reference = text.lower().startswith("reference")
        self.source_path_edit.setEnabled(is_reference)
        self.source_browse_btn.setEnabled(is_reference)

    def _on_browse_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.aac *.flac *.m4a *.ogg)",
        )
        if path:
            self.source_path_edit.setText(path)

    def _on_mode_changed(self, text: str) -> None:
        mode_map = {
            "Onset (Fast Cuts)": "onset",
            "Beat (Regular Cuts)": "beat",
            "Downbeat (Slow Cuts)": "downbeat",
            "Segment (Intro/Chorus)": "segment",
        }
        Config.set("beat_detector_mode", mode_map.get(text, "onset"))

    def _on_onset_threshold_changed(self, value: int) -> None:
        threshold = value / 100.0
        Config.set("beat_detector_onset_threshold", float(threshold))
        self.onset_threshold_label.setText(f"{threshold:.2f}")

    def _on_downbeat_weight_changed(self, value: int) -> None:
        weight = value / 100.0
        Config.set("beat_detector_downbeat_weight", float(weight))
        self.downbeat_weight_label.setText(f"{weight:.2f}")

    def _on_broll_mode_changed(self, text: str) -> None:
        if text.lower().startswith("transcript"):
            Config.set("beat_broll_mode", "transcript")
        else:
            Config.set("beat_broll_mode", "simple_random")

    def _on_broll_fill_changed(self, value: int) -> None:
        ratio = max(0.0, min(1.0, value / 100.0))
        Config.set("beat_broll_fill_ratio", float(ratio))
        self.broll_fill_label.setText(f"{ratio:.2f}")

    def _on_detect_beats(self) -> None:
        try:
            audio_path = self._resolve_audio_source()
        except Exception as exc:
            self.results.setHtml(f"<b>Error:</b> {exc}")
            return

        mode = str(Config.get("beat_detector_mode", "onset"))
        hop_length = int(Config.get("beat_detector_hop_length", 512))
        onset_threshold = float(Config.get("beat_detector_onset_threshold", 0.5))
        min_bpm = float(Config.get("beat_detector_min_bpm", 60))
        max_bpm = float(Config.get("beat_detector_max_bpm", 180))

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.detect_btn.setEnabled(False)

        self.worker = BeatWorker(
            audio_path=audio_path,
            mode=mode,
            hop_length=hop_length,
            onset_threshold=onset_threshold,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
        )
        self.worker.result_ready.connect(self._on_beats_ready)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.start()

    def _on_beats_ready(self, data: Dict[str, Any]) -> None:
        self.progress_bar.setVisible(False)
        self.detect_btn.setEnabled(True)
        self.beat_data = data if isinstance(data, dict) else None

        if not self.beat_data:
            self.results.setHtml("<b>Error:</b> Beat detection returned empty data.")
            return

        bpm = float(self.beat_data.get("bpm", 0.0) or 0.0)
        beats = self.beat_data.get("beats") or []
        onsets = self.beat_data.get("onsets") or []
        downbeats = self.beat_data.get("downbeats") or []
        duration = float(self.beat_data.get("duration", 0.0) or 0.0)

        html = "<h3>Beat Detection Results</h3>"
        html += f"<p><b>BPM:</b> {bpm:.2f}</p>"
        html += f"<p><b>Beats:</b> {len(beats)}</p>"
        html += f"<p><b>Onsets:</b> {len(onsets)}</p>"
        html += f"<p><b>Downbeats:</b> {len(downbeats)}</p>"
        html += f"<p><b>Duration:</b> {duration:.2f}s</p>"
        self.results.setHtml(html)

    def _on_error(self, message: str) -> None:
        self.progress_bar.setVisible(False)
        self.detect_btn.setEnabled(True)
        self.results.setHtml(f"<b>Error:</b> {message}")

    def _on_add_markers(self) -> None:
        if not self.beat_data:
            self.results.setHtml("<b>Error:</b> Run Detect Beats first.")
            return
        beats = self.beat_data.get("beats") or []
        onsets = self.beat_data.get("onsets") or []
        downbeats = self.beat_data.get("downbeats") or []
        beat_source = "beats"
        downbeat_source = "downbeats"
        if not beats and onsets:
            beats = onsets
            beat_source = "onsets"
        if self.use_downbeat_markers.isChecked() and not downbeats and beats:
            downbeats = [beats[idx] for idx in range(0, len(beats), 4)]
            downbeat_source = "derived"
        if not beats:
            self.results.setHtml("<b>Error:</b> No beats/onsets detected.")
            return
        logger.info(
            "Beat marker source: beats=%s (%s) downbeats=%s (%s)",
            len(beats),
            beat_source,
            len(downbeats),
            downbeat_source,
        )
        marker_color = str(Config.get("beat_marker_color", "blue"))
        marker_name = str(Config.get("beat_marker_name", "Beat"))
        filtered_beats = list(beats)
        downbeat_ok = True
        if self.use_downbeat_markers.isChecked():
            downbeat_color = str(Config.get("beat_downbeat_marker_color", "red"))
            downbeat_name = str(Config.get("beat_downbeat_marker_name", "Downbeat"))
            fps = float(self.resolve_api.get_timeline_fps() or 0.0)
            if fps > 0 and downbeats:
                downbeat_frames = {int(round(float(t) * fps)) for t in downbeats}
                filtered_beats = [
                    t
                    for t in beats
                    if int(round(float(t) * fps)) not in downbeat_frames
                ]
            downbeat_ok = self.resolve_api.add_downbeat_markers(
                downbeats,
                downbeat_color,
                downbeat_name,
                base_frame=self.beat_base_frame,
            )
        ok = self.resolve_api.add_beat_markers(
            filtered_beats,
            marker_color,
            marker_name,
            base_frame=self.beat_base_frame,
        )
        status = (
            f"Beat markers: {len(beats)} ({beat_source})."
            if ok
            else f"No beat markers added. ({len(beats)} {beat_source})"
        )
        if self.use_downbeat_markers.isChecked():
            status += (
                f" Downbeats: {len(downbeats)} ({downbeat_source})."
                if downbeat_ok
                else f" Downbeats not added. ({len(downbeats)} {downbeat_source})"
            )
        self.results.setHtml(f"<b>{status}</b>")

    def _on_generate_broll(self) -> None:
        if not self.beat_data:
            self.results.setHtml("<b>Error:</b> Run Detect Beats first.")
            return
        if self.broll_mode_combo.currentText().lower().startswith("transcript"):
            self.results.setHtml(
                "<b>Placeholder:</b> Transcript-aware B-roll will be wired in Phase 12-C."
            )
            return

        intervals, source_label = self._get_beat_intervals()
        if not intervals:
            self.results.setHtml("<b>Error:</b> No beat intervals available.")
            return
        intervals_total = len(intervals)

        pool = self._build_selected_clip_pool()
        if not pool:
            self.results.setHtml("<b>Error:</b> Select timeline video clips to use as B-roll.")
            return

        fps = float(self.resolve_api.get_timeline_fps() or 0.0)
        if fps <= 0:
            self.results.setHtml("<b>Error:</b> Invalid timeline FPS.")
            return

        timeline = self.resolve_api.get_current_timeline()
        track_count = int(timeline.GetTrackCount("video") or 0)
        target_track = max(1, track_count + 1)

        fill_ratio = float(Config.get("beat_broll_fill_ratio", 1.0) or 0.0)
        fill_ratio = max(0.0, min(1.0, fill_ratio))
        target_count = int(round(intervals_total * fill_ratio))
        if target_count <= 0:
            self.results.setHtml("<b>Error:</b> B-roll fill ratio yields no intervals.")
            return

        seed = self._resolve_broll_seed()
        rng = random.Random(seed)
        pool_selected = list(pool)
        max_count = min(len(pool_selected), target_count)
        if max_count <= 0:
            self.results.setHtml("<b>Error:</b> No usable B-roll clips in selection.")
            return

        intervals_selected = intervals
        if max_count < intervals_total:
            indices = rng.sample(range(intervals_total), max_count)
            intervals_selected = [intervals[i] for i in sorted(indices)]
        if len(pool_selected) > max_count:
            pool_selected = [pool_selected[i] for i in rng.sample(range(len(pool_selected)), max_count)]
        rng.shuffle(pool_selected)

        if max_count < intervals_total or max_count < len(pool):
            logger.info(
                "Beat B-roll fill: ratio=%.2f total=%s target=%s pool=%s selected=%s",
                fill_ratio,
                intervals_total,
                target_count,
                len(pool),
                max_count,
            )

        inserted = self._place_clips_on_intervals(
            pool_selected,
            intervals_selected,
            fps,
            target_track,
            rng=None,
            randomize=False,
        )

        status = (
            f"Beat B-roll placed on V{target_track}: {inserted}/{len(intervals_selected)} intervals "
            f"({source_label}, seed={seed}, fill={fill_ratio:.2f}"
        )
        if len(intervals_selected) != intervals_total:
            status += f", total={intervals_total}"
        status += ")."
        self.results.setHtml(f"<b>{status}</b>")

    def _on_auto_edit(self) -> None:
        if not self.beat_data:
            self.results.setHtml("<b>Error:</b> Run Detect Beats first.")
            return

        intervals, source_label = self._get_beat_intervals()
        if not intervals:
            self.results.setHtml("<b>Error:</b> No beat intervals available.")
            return

        pool = self._build_selected_clip_pool()
        if not pool:
            self.results.setHtml("<b>Error:</b> Select timeline video clips to auto-edit.")
            return

        fps = float(self.resolve_api.get_timeline_fps() or 0.0)
        if fps <= 0:
            self.results.setHtml("<b>Error:</b> Invalid timeline FPS.")
            return

        timeline = self.resolve_api.get_current_timeline()
        track_count = int(timeline.GetTrackCount("video") or 0)
        target_track = max(1, track_count + 1)

        inserted = self._place_clips_on_intervals(
            pool,
            intervals,
            fps,
            target_track,
            rng=None,
            randomize=False,
        )

        status = (
            f"Auto-edit placed on V{target_track}: {inserted}/{len(intervals)} intervals "
            f"({source_label})."
        )
        self.results.setHtml(f"<b>{status}</b>")

    def _get_beat_intervals(self) -> Tuple[List[Tuple[float, float]], str]:
        if not self.beat_data:
            return [], "beats"
        beats = self.beat_data.get("beats") or []
        onsets = self.beat_data.get("onsets") or []
        downbeats = self.beat_data.get("downbeats") or []
        duration = float(self.beat_data.get("duration", 0.0) or 0.0)

        mode = str(Config.get("beat_detector_mode", "onset")).strip().lower()
        source_label = "beats"
        if mode == "downbeat" and downbeats:
            beats = downbeats
            source_label = "downbeats"
        elif not beats and onsets:
            beats = onsets
            source_label = "onsets"

        beat_times = sorted(float(x) for x in beats if x is not None)
        if not beat_times:
            return [], source_label

        intervals: List[Tuple[float, float]] = []
        for idx, start in enumerate(beat_times):
            end = beat_times[idx + 1] if idx + 1 < len(beat_times) else duration
            if end <= start:
                continue
            intervals.append((float(start), float(end)))
        return intervals, source_label

    def _resolve_broll_seed(self) -> int:
        text = self.broll_seed_edit.text().strip()
        if text:
            try:
                return int(text)
            except Exception:
                return int(zlib.adler32(text.encode("utf-8")) & 0x7FFFFFFF)
        return int(random.SystemRandom().randint(0, 2**31 - 1))

    def _build_selected_clip_pool(self) -> List[Dict[str, Any]]:
        selected = self.resolve_api.get_selected_items(["video"])
        if not selected:
            return []
        fps = float(self.resolve_api.get_timeline_fps() or 0.0)
        pool: List[Dict[str, Any]] = []
        for item in selected:
            media_path = self.resolve_api.get_item_media_path(item)
            if not media_path:
                continue
            media_item = self.resolve_api.import_media_if_needed(str(media_path))
            if media_item is None:
                continue
            source_range = self.resolve_api.get_clip_source_range_seconds(item)
            if not source_range:
                source_range = self.resolve_api.get_item_range_seconds(item)
            if not source_range:
                continue
            source_in_sec, source_out_sec = source_range
            if source_out_sec <= source_in_sec:
                continue
            clip_fps = float(
                self.resolve_api._get_media_item_fps(media_item, fps)  # pylint: disable=protected-access
            )
            source_in_frame = int(round(source_in_sec * clip_fps))
            source_out_frame = int(round(source_out_sec * clip_fps))
            if source_out_frame <= source_in_frame:
                continue
            total_frames = int(
                self.resolve_api._get_media_item_duration_frames(media_item, clip_fps)  # pylint: disable=protected-access
            )
            if total_frames and source_out_frame > total_frames:
                source_out_frame = total_frames
                if source_out_frame <= source_in_frame:
                    continue
            pool.append(
                {
                    "media_item": media_item,
                    "clip_fps": clip_fps,
                    "source_in_frame": source_in_frame,
                    "source_out_frame": source_out_frame,
                    "media_path": str(media_path),
                }
            )
        return pool

    def _place_clips_on_intervals(
        self,
        pool: List[Dict[str, Any]],
        intervals: List[Tuple[float, float]],
        fps: float,
        target_track: int,
        rng: random.Random | None,
        randomize: bool,
    ) -> int:
        if not pool or not intervals:
            return 0
        base_frame = int(self.beat_base_frame or 0)
        inserted = 0
        for idx, (start, end) in enumerate(intervals):
            clip = rng.choice(pool) if (randomize and rng) else pool[idx % len(pool)]
            clip_fps = float(clip["clip_fps"])
            source_in = int(clip["source_in_frame"])
            source_out = int(clip["source_out_frame"])
            available_frames = max(0, source_out - source_in)
            if available_frames <= 0:
                continue
            interval_sec = max(0.0, float(end) - float(start))
            desired_frames = int(round(interval_sec * clip_fps))
            if desired_frames <= 0:
                continue
            use_frames = min(available_frames, desired_frames)
            if use_frames <= 0:
                continue
            record_frame = base_frame + int(round(float(start) * fps))
            inserted_item = self.resolve_api.insert_clip_at_position_with_range(
                "video",
                int(target_track),
                clip["media_item"],
                int(record_frame),
                int(source_in),
                int(source_in + use_frames),
            )
            if inserted_item is not None:
                inserted += 1
        return inserted

    def _resolve_audio_source(self) -> Path:
        source = self.source_combo.currentText().lower()
        if source.startswith("reference"):
            self.beat_base_frame = None
            path = Path(self.source_path_edit.text().strip())
            if not path.exists():
                raise RuntimeError("Reference audio file not found.")
            return path

        selected = self.resolve_api.get_selected_items(["audio"])
        if not selected:
            selected = self.resolve_api.get_items_at_playhead(["audio"])
        if not selected:
            selected = self.resolve_api.get_selected_items(["video"])
        if not selected:
            selected = self.resolve_api.get_items_at_playhead(["video"])
        if not selected:
            raise RuntimeError("No audio clip selected or under playhead.")

        clip = selected[0]
        self.beat_base_frame = None
        timeline_start = int(self.resolve_api.get_timeline_start_frame() or 0)
        clip_start = self.resolve_api.get_item_start_frame(clip)
        if clip_start is not None:
            base_frame = int(clip_start)
            if timeline_start and base_frame >= timeline_start:
                base_frame -= timeline_start
            if base_frame < 0:
                base_frame = 0
            self.beat_base_frame = base_frame
            logger.info(
                "Beat marker base frame: %s (timeline_start=%s)",
                base_frame,
                timeline_start,
            )
        temp_path = Path(tempfile.gettempdir()) / "videoforge_timeline_audio.wav"
        start_sec = None
        end_sec = None
        source_range = self.resolve_api.get_clip_source_range_seconds(clip)
        if source_range:
            start_sec, end_sec = source_range
        elif self.resolve_api.get_item_range_seconds(clip):
            start_sec, end_sec = self.resolve_api.get_item_range_seconds(clip)
        if start_sec is not None and end_sec is not None and end_sec <= start_sec:
            start_sec, end_sec = None, None
        if start_sec is not None and end_sec is not None:
            logger.info(
                "Beat source range: %.2fs-%.2fs (%s)",
                float(start_sec),
                float(end_sec),
                temp_path,
            )
        return Path(
            self.resolve_api.export_clip_audio(
                clip,
                str(temp_path),
                start_sec=start_sec,
                end_sec=end_sec,
            )
        )

    @staticmethod
    def _make_slider(min_val: int, max_val: int, current: int) -> QSlider:
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(min_val))
        slider.setMaximum(int(max_val))
        slider.setValue(int(current))
        return slider
