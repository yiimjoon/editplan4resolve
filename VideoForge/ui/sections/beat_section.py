"""Audio beat engine UI section."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

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

        self.min_bpm_slider = self._make_slider(40, 240, int(Config.get("beat_detector_min_bpm", 60)))
        self.min_bpm_label = QLabel(f"{self.min_bpm_slider.value()} BPM")
        self.min_bpm_slider.valueChanged.connect(self._on_min_bpm_changed)
        settings_layout.addRow("Min BPM:", self.min_bpm_slider)
        settings_layout.addRow("", self.min_bpm_label)

        self.max_bpm_slider = self._make_slider(60, 300, int(Config.get("beat_detector_max_bpm", 180)))
        self.max_bpm_label = QLabel(f"{self.max_bpm_slider.value()} BPM")
        self.max_bpm_slider.valueChanged.connect(self._on_max_bpm_changed)
        settings_layout.addRow("Max BPM:", self.max_bpm_slider)
        settings_layout.addRow("", self.max_bpm_label)

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

    def _on_min_bpm_changed(self, value: int) -> None:
        max_val = self.max_bpm_slider.value()
        if value >= max_val:
            value = max(40, max_val - 1)
            self.min_bpm_slider.blockSignals(True)
            self.min_bpm_slider.setValue(value)
            self.min_bpm_slider.blockSignals(False)
        Config.set("beat_detector_min_bpm", int(value))
        self.min_bpm_label.setText(f"{value} BPM")

    def _on_max_bpm_changed(self, value: int) -> None:
        min_val = self.min_bpm_slider.value()
        if value <= min_val:
            value = min(300, min_val + 1)
            self.max_bpm_slider.blockSignals(True)
            self.max_bpm_slider.setValue(value)
            self.max_bpm_slider.blockSignals(False)
        Config.set("beat_detector_max_bpm", int(value))
        self.max_bpm_label.setText(f"{value} BPM")

    def _on_onset_threshold_changed(self, value: int) -> None:
        threshold = value / 100.0
        Config.set("beat_detector_onset_threshold", float(threshold))
        self.onset_threshold_label.setText(f"{threshold:.2f}")

    def _on_downbeat_weight_changed(self, value: int) -> None:
        weight = value / 100.0
        Config.set("beat_detector_downbeat_weight", float(weight))
        self.downbeat_weight_label.setText(f"{weight:.2f}")

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
        downbeats = self.beat_data.get("downbeats") or []
        marker_color = str(Config.get("beat_marker_color", "blue"))
        marker_name = str(Config.get("beat_marker_name", "Beat"))
        ok = self.resolve_api.add_beat_markers(beats, marker_color, marker_name)
        downbeat_ok = True
        if self.use_downbeat_markers.isChecked():
            downbeat_color = str(Config.get("beat_downbeat_marker_color", "red"))
            downbeat_name = str(Config.get("beat_downbeat_marker_name", "Downbeat"))
            downbeat_ok = self.resolve_api.add_downbeat_markers(
                downbeats,
                downbeat_color,
                downbeat_name,
            )
        status = "Added beat markers." if ok else "No beat markers added."
        if self.use_downbeat_markers.isChecked():
            status += " Downbeats added." if downbeat_ok else " Downbeats not added."
        self.results.setHtml(f"<b>{status}</b>")

    def _on_generate_broll(self) -> None:
        self.results.setHtml(
            "<b>Placeholder:</b> Beat-based B-roll generation will be wired in Phase 12-C."
        )

    def _on_auto_edit(self) -> None:
        self.results.setHtml(
            "<b>Placeholder:</b> Auto-edit to beat will be wired in Phase 12-C."
        )

    def _resolve_audio_source(self) -> Path:
        source = self.source_combo.currentText().lower()
        if source.startswith("reference"):
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
            raise RuntimeError("No audio clip selected or under playhead.")

        clip = selected[0]
        temp_path = Path(tempfile.gettempdir()) / "videoforge_timeline_audio.wav"
        return Path(self.resolve_api.export_clip_audio(clip, str(temp_path)))

    @staticmethod
    def _make_slider(min_val: int, max_val: int, current: int) -> QSlider:
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(min_val))
        slider.setMaximum(int(max_val))
        slider.setValue(int(current))
        return slider
