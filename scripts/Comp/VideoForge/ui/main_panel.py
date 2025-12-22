from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

import yaml

from VideoForge.broll.db import LibraryDB
from VideoForge.broll.indexer import scan_library
from VideoForge.broll.sidecar_tools import generate_comfyui_sidecars, generate_scene_sidecars
from VideoForge.config.config_manager import Config
from VideoForge.plugin.resolve_bridge import ResolveBridge

try:
    from PySide6.QtCore import QObject, QThread, Signal, Qt, QTimer
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QFileDialog,
        QFrame,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QProgressBar,
        QSlider,
        QVBoxLayout,
        QWidget,
    )
except Exception:
    try:
        from PySide2.QtCore import QObject, QThread, Signal, Qt, QTimer
        from PySide2.QtWidgets import (
            QApplication,
            QComboBox,
            QFileDialog,
            QFrame,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QPushButton,
            QProgressBar,
            QSlider,
            QVBoxLayout,
            QWidget,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PySide6 or PySide2 is required for the VideoForge panel.") from exc


# --- Color Palette (DaVinci Resolve Style) ---
COLORS = {
    "bg_window": "#1c1c1c",
    "bg_card": "#2a2a2a",
    "text_main": "#ececec",
    "text_dim": "#9e9e9e",
    "accent": "#da8a35",
    "accent_hover": "#e69b4e",
    "button_bg": "#3e3e3e",
    "button_hover": "#4e4e4e",
    "border": "#121212",
    "success": "#4caf50",
    "error": "#f44336",
}

VERSION = "v1.3.2"

# --- Stylesheet ---
STYLESHEET = f"""
    QWidget {{
        background-color: {COLORS['bg_window']};
        font-family: 'Segoe UI', 'Open Sans', sans-serif;
        font-size: 13px;
        color: {COLORS['text_main']};
    }}

    /* Section Cards */
    QFrame#SectionCard {{
        background-color: {COLORS['bg_card']};
        border-radius: 8px;
        border: 1px solid {COLORS['border']};
        padding: 12px;
    }}

    /* Section Titles */
    QLabel#SectionTitle {{
        font-size: 14px;
        font-weight: bold;
        color: {COLORS['accent']};
        padding-bottom: 4px;
    }}

    /* Description Labels */
    QLabel#Description {{
        color: {COLORS['text_dim']};
        margin-bottom: 6px;
    }}

    /* Status Labels */
    QLabel#StatusLabel {{
        color: {COLORS['text_dim']};
        margin-top: 6px;
    }}

    /* Buttons */
    QPushButton {{
        background-color: {COLORS['button_bg']};
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        color: {COLORS['text_main']};
        font-weight: 500;
    }}
    QPushButton:hover {{
        background-color: {COLORS['button_hover']};
    }}
    QPushButton:pressed {{
        background-color: {COLORS['border']};
    }}
    QPushButton:disabled {{
        background-color: {COLORS['border']};
        color: {COLORS['text_dim']};
    }}

    /* Primary Action Button (Orange) */
    QPushButton#PrimaryButton {{
        background-color: {COLORS['accent']};
        color: #1a1a1a;
        font-weight: bold;
    }}
    QPushButton#PrimaryButton:hover {{
        background-color: {COLORS['accent_hover']};
    }}
    QPushButton#PrimaryButton:pressed {{
        background-color: #c67a2a;
    }}

    /* Text Inputs */
    QLineEdit {{
        background-color: {COLORS['border']};
        border: 1px solid {COLORS['button_bg']};
        border-radius: 4px;
        padding: 6px 8px;
        color: {COLORS['text_main']};
        selection-background-color: {COLORS['accent']};
    }}
    QLineEdit:focus {{
        border: 1px solid {COLORS['accent']};
    }}

    /* Sliders */
    QSlider::groove:horizontal {{
        border: none;
        height: 6px;
        background: {COLORS['border']};
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {COLORS['text_main']};
        border: 2px solid {COLORS['bg_window']};
        width: 14px;
        height: 14px;
        margin: -5px 0;
        border-radius: 8px;
    }}
    QSlider::handle:horizontal:hover {{
        background: {COLORS['accent']};
    }}
    QSlider::sub-page:horizontal {{
        background: {COLORS['accent']};
        border-radius: 3px;
    }}

    /* Progress Bar */
    QProgressBar {{
        border: none;
        background-color: {COLORS['border']};
        border-radius: 2px;
        text-align: center;
        max-height: 4px;
    }}
    QProgressBar::chunk {{
        background-color: {COLORS['accent']};
        border-radius: 2px;
    }}

    /* Separator Line */
    QFrame#Separator {{
        background-color: {COLORS['button_bg']};
        max-height: 1px;
    }}
"""


class Worker(QObject):
    """Worker object for running tasks in a separate thread."""

    finished = Signal(object, object)

    def __init__(self, func, *args, **kwargs) -> None:
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self) -> None:
        logger = logging.getLogger("VideoForge.ui.Worker")
        try:
            logger.info("=== Worker thread started: %s ===", getattr(self.func, "__name__", "unknown"))
            result = self.func(*self.args, **self.kwargs)
            logger.info("=== Worker thread finished: %s ===", getattr(self.func, "__name__", "unknown"))
            self.finished.emit(result, None)
        except Exception as exc:
            logger.error("=== Worker thread error: %s ===", exc, exc_info=True)
            self.finished.emit(None, exc)


DEFAULT_SETTINGS = {
    "silence": {
        "enable_removal": False,
        "threshold_db": -34,
        "min_keep_duration": 0.4,
        "buffer_before": 0.2,
        "buffer_after": 0.3,
    },
    "broll": {
        "matching_threshold": 0.65,
        "max_overlay_duration": 3.0,
        "fit_mode": "center",
        "crossfade": 0.3,
        "cooldown_sec": 30,
        "hybrid_weights": {"vector": 0.5, "text": 0.5, "prior": 0.1},
        "visual_similarity_threshold": 0.85,
        "cooldown_penalty": 0.4,
    },
    "whisper": {"task": "translate", "language": "auto"},
    "query": {"top_n_tokens": 6},
    "vision": {"max_frames": 12, "base_frames": 6},
    "resolve": {"timeline_name_prefix": "VideoForge_"},
}


class VideoForgePanel(QWidget):
    """Modern styled panel for VideoForge plugin."""

    status_signal = Signal(str)
    progress_signal = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self._startup_warning: Optional[str] = None
        self.settings = self._load_settings()
        self.bridge = ResolveBridge(self.settings)
        self.saved_broll_dir = Config.get("broll_dir")
        self.project_db: Optional[str] = None
        self.main_video_path: Optional[str] = None
        self._threads: list[QThread] = []
        self._threads_lock = threading.Lock()
        self.status_signal.connect(self._set_status)
        self.progress_signal.connect(self._set_progress_value)
        self._init_ui()
        if self._startup_warning:
            self._set_status(self._startup_warning)
        self._restore_library_path()

    def _init_ui(self) -> None:
        """Initialize the modernized UI."""
        self.setStyleSheet(STYLESHEET)
        self.setWindowTitle("VideoForge - Auto B-roll")
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # --- Header ---
        self._setup_header(layout)

        # --- Step 1: Analyze ---
        self._setup_step1_analyze(layout)

        # --- Step 2: Library ---
        self._setup_step2_library(layout)

        # --- Step 3: Match & Apply ---
        self._setup_step3_match(layout)

        # --- Step 4: Advanced ---
        self._setup_step4_advanced(layout)

        # --- Footer ---
        self._setup_footer(layout)

    def _setup_header(self, parent_layout: QVBoxLayout) -> None:
        """Create the header with title and subtitle."""
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 4)

        title = QLabel("VideoForge")
        title.setStyleSheet(
            f"font-size: 18px; font-weight: 800; color: {COLORS['text_main']}; "
            "letter-spacing: 1px; background: transparent;"
        )

        subtitle = QLabel("AI B-roll Automation")
        subtitle.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-size: 11px; background: transparent;"
        )
        subtitle.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        version = QLabel(VERSION)
        version.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-size: 10px; background: transparent;"
        )
        version.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(subtitle)
        header_layout.addSpacing(8)
        header_layout.addWidget(version)
        parent_layout.addLayout(header_layout)

        # Separator line
        line = QFrame()
        line.setObjectName("Separator")
        line.setFrameShape(QFrame.HLine)
        line.setFixedHeight(1)
        parent_layout.addWidget(line)

    def _setup_step1_analyze(self, parent_layout: QVBoxLayout) -> None:
        """Step 1: Main Clip Analysis Section."""
        card = self._create_card()
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(8)

        # Title
        card_layout.addWidget(self._create_section_title("1. Analyze Main Clip"))

        # Description
        desc = QLabel("Select a clip in the timeline, then analyze its content.")
        desc.setObjectName("Description")
        desc.setWordWrap(True)
        card_layout.addWidget(desc)

        # Button
        self.analyze_btn = QPushButton("Analyze Selected Clip")
        self.analyze_btn.setObjectName("PrimaryButton")
        self.analyze_btn.setCursor(Qt.PointingHandCursor)
        self.analyze_btn.setFixedHeight(36)
        self.analyze_btn.clicked.connect(self._on_analyze_clicked)
        card_layout.addWidget(self.analyze_btn)

        # Status
        self.analyze_status = QLabel("Status: Not analyzed")
        self.analyze_status.setObjectName("StatusLabel")
        card_layout.addWidget(self.analyze_status)

        log_row = QHBoxLayout()
        log_row.setSpacing(6)
        self.log_path_label = QLabel("Log: VideoForge.log")
        log_path = self._get_log_path()
        if log_path:
            self.log_path_label.setToolTip(str(log_path))
        self.log_path_label.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-size: 11px; background: transparent;"
        )
        self.open_log_btn = QPushButton("Open Log File")
        self.open_log_btn.setCursor(Qt.PointingHandCursor)
        self.open_log_btn.clicked.connect(self._on_open_log)
        log_row.addWidget(self.log_path_label)
        log_row.addStretch()
        log_row.addWidget(self.open_log_btn)
        card_layout.addLayout(log_row)

        parent_layout.addWidget(card)

    def _setup_step2_library(self, parent_layout: QVBoxLayout) -> None:
        """Step 2: B-roll Library Section."""
        card = self._create_card()
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(8)

        # Title
        card_layout.addWidget(self._create_section_title("2. B-roll Library"))

        # Path input row
        path_layout = QHBoxLayout()
        path_layout.setSpacing(6)

        self.library_path_edit = QLineEdit()
        self.library_path_edit.setPlaceholderText("Library DB path...")

        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setFixedWidth(36)
        self.browse_btn.setToolTip("Browse for library DB file")
        self.browse_btn.setCursor(Qt.PointingHandCursor)
        self.browse_btn.clicked.connect(self._on_browse_library)

        path_layout.addWidget(self.library_path_edit)
        path_layout.addWidget(self.browse_btn)
        card_layout.addLayout(path_layout)

        # Scan button
        self.scan_library_btn = QPushButton("Scan Library Folder")
        self.scan_library_btn.setCursor(Qt.PointingHandCursor)
        self.scan_library_btn.clicked.connect(self._on_scan_library)
        card_layout.addWidget(self.scan_library_btn)

        # Status
        self.library_status = QLabel("Status: Library not scanned")
        self.library_status.setObjectName("StatusLabel")
        card_layout.addWidget(self.library_status)

        parent_layout.addWidget(card)

    def _setup_step3_match(self, parent_layout: QVBoxLayout) -> None:
        """Step 3: Match & Apply Section."""
        card = self._create_card()
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(8)

        # Title
        card_layout.addWidget(self._create_section_title("3. Match & Apply"))

        # Threshold slider header
        slider_header = QHBoxLayout()
        slider_label = QLabel("Matching Threshold")
        slider_label.setStyleSheet("background: transparent;")

        self.threshold_label = QLabel(
            f"{int(self.settings['broll']['matching_threshold'] * 100)}%"
        )
        self.threshold_label.setStyleSheet(
            f"color: {COLORS['accent']}; font-weight: bold; background: transparent;"
        )
        self.threshold_label.setAlignment(Qt.AlignRight)

        slider_header.addWidget(slider_label)
        slider_header.addWidget(self.threshold_label)
        card_layout.addLayout(slider_header)

        # Slider
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(
            int(self.settings["broll"]["matching_threshold"] * 100)
        )
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        card_layout.addWidget(self.threshold_slider)

        card_layout.addSpacing(6)

        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)

        self.match_btn = QPushButton("Match B-roll")
        self.match_btn.setCursor(Qt.PointingHandCursor)
        self.match_btn.clicked.connect(self._on_match_clicked)

        self.apply_btn = QPushButton("Apply to Timeline")
        self.apply_btn.setObjectName("PrimaryButton")
        self.apply_btn.setCursor(Qt.PointingHandCursor)
        self.apply_btn.clicked.connect(self._on_apply_clicked)

        button_layout.addWidget(self.match_btn, 1)
        button_layout.addWidget(self.apply_btn, 2)
        card_layout.addLayout(button_layout)

        parent_layout.addWidget(card)

    def _setup_step4_advanced(self, parent_layout: QVBoxLayout) -> None:
        """Step 4: Advanced Matching + Sidecar Tools."""
        card = self._create_card()
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(8)

        card_layout.addWidget(self._create_section_title("4. Advanced"))

        card_layout.addWidget(QLabel("Whisper Language"))
        self.language_combo = QComboBox()
        self.language_combo.addItems(["auto", "ko", "en"])
        current_lang = self.settings.get("whisper", {}).get("language", "auto")
        if current_lang in {"auto", "ko", "en"}:
            self.language_combo.setCurrentText(current_lang)
        else:
            self.language_combo.setCurrentText("auto")
        self.language_combo.currentTextChanged.connect(self._on_language_changed)
        card_layout.addWidget(self.language_combo)

        srt_row = QHBoxLayout()
        srt_row.setSpacing(6)
        self.srt_path_edit = QLineEdit()
        self.srt_path_edit.setPlaceholderText("Optional SRT path...")
        self.srt_browse_btn = QPushButton("Browse")
        self.srt_browse_btn.setFixedWidth(72)
        self.srt_browse_btn.setCursor(Qt.PointingHandCursor)
        self.srt_browse_btn.clicked.connect(self._on_browse_srt)
        srt_row.addWidget(self.srt_path_edit)
        srt_row.addWidget(self.srt_browse_btn)
        card_layout.addLayout(srt_row)

        card_layout.addWidget(QLabel("Hybrid Weights"))
        vector_weight = float(self.settings["broll"]["hybrid_weights"]["vector"])
        text_weight = float(self.settings["broll"]["hybrid_weights"]["text"])
        self.vector_weight_label = QLabel(
            f"Vector: {vector_weight:.2f} / Text: {text_weight:.2f}"
        )
        self.vector_weight_label.setStyleSheet(
            f"color: {COLORS['text_dim']}; background: transparent;"
        )
        card_layout.addWidget(self.vector_weight_label)

        self.vector_weight_slider = QSlider(Qt.Horizontal)
        self.vector_weight_slider.setMinimum(0)
        self.vector_weight_slider.setMaximum(100)
        self.vector_weight_slider.setValue(
            int(self.settings["broll"]["hybrid_weights"]["vector"] * 100)
        )
        self.vector_weight_slider.valueChanged.connect(self._on_vector_weight_changed)
        card_layout.addWidget(self.vector_weight_slider)

        prior_weight = float(self.settings["broll"]["hybrid_weights"]["prior"])
        self.prior_weight_label = QLabel(f"Prior Weight: {prior_weight:.2f}")
        self.prior_weight_label.setStyleSheet(
            f"color: {COLORS['text_dim']}; background: transparent;"
        )
        card_layout.addWidget(self.prior_weight_label)

        self.prior_weight_slider = QSlider(Qt.Horizontal)
        self.prior_weight_slider.setMinimum(0)
        self.prior_weight_slider.setMaximum(30)
        self.prior_weight_slider.setValue(
            int(self.settings["broll"]["hybrid_weights"]["prior"] * 100)
        )
        self.prior_weight_slider.valueChanged.connect(self._on_prior_weight_changed)
        card_layout.addWidget(self.prior_weight_slider)

        visual_threshold = float(self.settings["broll"]["visual_similarity_threshold"])
        self.visual_threshold_label = QLabel(f"Visual Filter: {visual_threshold:.2f}")
        self.visual_threshold_label.setStyleSheet(
            f"color: {COLORS['text_dim']}; background: transparent;"
        )
        card_layout.addWidget(self.visual_threshold_label)

        self.visual_threshold_slider = QSlider(Qt.Horizontal)
        self.visual_threshold_slider.setMinimum(70)
        self.visual_threshold_slider.setMaximum(95)
        self.visual_threshold_slider.setValue(
            int(self.settings["broll"]["visual_similarity_threshold"] * 100)
        )
        self.visual_threshold_slider.valueChanged.connect(self._on_visual_threshold_changed)
        card_layout.addWidget(self.visual_threshold_slider)

        self.sidecar_btn = QPushButton("Generate Scene Sidecars")
        self.sidecar_btn.setCursor(Qt.PointingHandCursor)
        self.sidecar_btn.clicked.connect(self._on_generate_sidecars)
        card_layout.addWidget(self.sidecar_btn)

        self.comfyui_sidecar_btn = QPushButton("Generate ComfyUI Sidecars")
        self.comfyui_sidecar_btn.setCursor(Qt.PointingHandCursor)
        self.comfyui_sidecar_btn.clicked.connect(self._on_generate_comfyui_sidecars)
        card_layout.addWidget(self.comfyui_sidecar_btn)

        parent_layout.addWidget(card)

    def _setup_footer(self, parent_layout: QVBoxLayout) -> None:
        """Footer with progress bar and global status."""
        parent_layout.addStretch()

        # Progress bar (hidden by default)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # Indeterminate mode
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        parent_layout.addWidget(self.progress)

        # Global status
        self.status = QLabel("Ready")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-size: 11px; background: transparent;"
        )
        parent_layout.addWidget(self.status)

        self.last_log_label = QLabel("")
        self.last_log_label.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-size: 10px; "
            "font-family: 'Consolas', monospace; background: transparent;"
        )
        self.last_log_label.setWordWrap(True)
        parent_layout.addWidget(self.last_log_label)

        self.log_timer = QTimer(self)
        self.log_timer.timeout.connect(self._update_last_log)
        self.log_timer.start(1000)

    # --- Helper Methods ---

    def _create_card(self) -> QFrame:
        """Create a styled card frame."""
        frame = QFrame()
        frame.setObjectName("SectionCard")
        return frame

    def _create_section_title(self, text: str) -> QLabel:
        """Create a styled section title label."""
        label = QLabel(text)
        label.setObjectName("SectionTitle")
        return label

    def _load_settings(self) -> dict:
        settings_path = self._resolve_repo_root() / "VideoForge" / "config" / "settings.yaml"
        if not settings_path.exists():
            self._startup_warning = "Settings file missing. Using defaults."
            return DEFAULT_SETTINGS.copy()
        try:
            data = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
        except Exception:
            self._startup_warning = "Failed to read settings.yaml. Using defaults."
            return DEFAULT_SETTINGS.copy()
        merged = DEFAULT_SETTINGS.copy()
        for key, value in data.items():
            if isinstance(value, dict) and key in merged:
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _resolve_repo_root() -> Path:
        script_root = Path(__file__).resolve().parents[2]
        if (script_root / "VideoForge").exists():
            return script_root
        env_root = os.environ.get("VIDEOFORGE_ROOT")
        if env_root:
            env_path = Path(env_root)
            if env_path.exists():
                return env_path
        root = Path(__file__).resolve().parents[3]
        if not root.exists():
            raise RuntimeError("Failed to resolve VideoForge root path.")
        return root

    def _set_status(self, message: str) -> None:
        self.status.setText(message)

    def _update_status_safe(self, message: str) -> None:
        logging.getLogger(__name__).info(message)
        self.status_signal.emit(message)

    def _get_log_path(self) -> Optional[Path]:
        env_path = os.environ.get("VIDEOFORGE_LOG_PATH")
        if env_path:
            return Path(env_path)
        appdata = os.environ.get("APPDATA")
        if not appdata:
            return None
        return (
            Path(appdata)
            / "Blackmagic Design"
            / "DaVinci Resolve"
            / "Support"
            / "VideoForge.log"
        )

    def _set_busy(self, busy: bool) -> None:
        self.progress.setVisible(busy)
        self.analyze_btn.setEnabled(not busy)
        self.scan_library_btn.setEnabled(not busy)
        self.match_btn.setEnabled(not busy)
        self.apply_btn.setEnabled(not busy)

    def _set_progress_value(self, value: int) -> None:
        try:
            self.progress.setRange(0, 100)
            self.progress.setValue(max(0, min(100, int(value))))
            self.progress.setTextVisible(False)
        except Exception:
            return

    def _update_progress_safe(self, value: int) -> None:
        self.progress_signal.emit(value)

    def _run_worker(self, func, *args, on_done=None) -> None:
        self._set_busy(True)
        thread = QThread(self)
        worker = Worker(func, *args)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(lambda result, err: self._on_worker_done(thread, result, err, on_done))
        thread.start()
        with self._threads_lock:
            self._threads.append(thread)

    def _on_worker_done(self, thread: QThread, result, err, on_done) -> None:
        self._set_busy(False)
        if err:
            self._set_status(f"Error: {err}")
            self.status.setStyleSheet(
                f"color: {COLORS['error']}; font-size: 11px; background: transparent;"
            )
        elif on_done:
            on_done(result)
            self.status.setStyleSheet(
                f"color: {COLORS['text_dim']}; font-size: 11px; background: transparent;"
            )
        thread.quit()
        thread.wait()
        with self._threads_lock:
            if thread in self._threads:
                self._threads.remove(thread)

    def _on_threshold_changed(self, value: int) -> None:
        self.threshold_label.setText(f"{value}%")
        # Color feedback based on value
        if value < 30:
            color = COLORS["text_dim"]
        elif value > 80:
            color = COLORS["success"]
        else:
            color = COLORS["accent"]
        self.threshold_label.setStyleSheet(
            f"color: {color}; font-weight: bold; background: transparent;"
        )

    def _on_vector_weight_changed(self, value: int) -> None:
        vector_weight = value / 100.0
        text_weight = 1.0 - vector_weight
        self.settings["broll"]["hybrid_weights"]["vector"] = vector_weight
        self.settings["broll"]["hybrid_weights"]["text"] = text_weight
        self.vector_weight_label.setText(
            f"Vector: {vector_weight:.2f} / Text: {text_weight:.2f}"
        )

    def _on_prior_weight_changed(self, value: int) -> None:
        prior_weight = value / 100.0
        self.settings["broll"]["hybrid_weights"]["prior"] = prior_weight
        self.prior_weight_label.setText(f"Prior Weight: {prior_weight:.2f}")

    def _on_visual_threshold_changed(self, value: int) -> None:
        threshold = value / 100.0
        self.settings["broll"]["visual_similarity_threshold"] = threshold
        self.visual_threshold_label.setText(f"Visual Filter: {threshold:.2f}")

    def _on_language_changed(self, value: str) -> None:
        self.settings.setdefault("whisper", {})["language"] = value

    # --- Event Handlers ---

    def _on_analyze_clicked(self) -> None:
        def _analyze():
            logger = logging.getLogger("VideoForge.ui.Analyze")
            logger.info("[1/6] Reading settings from UI")
            self._update_progress_safe(8)
            self._update_status_safe("Step 1/6: Reading settings...")
            srt_path = self.srt_path_edit.text().strip() if self.srt_path_edit else ""
            srt_path = srt_path or None
            whisper_task = self.settings.get("whisper", {}).get("task", "translate")
            whisper_language = self.settings.get("whisper", {}).get("language", "auto")

            logger.info("[2/6] Starting analyze_selected_clip")
            self._update_progress_safe(16)
            self._update_status_safe(
                "Step 2/6: Loading Whisper model (first run may take 1-2 min)..."
            )

            result_container = [None]
            error_container = [None]

            def _analyze_with_timeout():
                try:
                    result_container[0] = self.bridge.analyze_selected_clip(
                        whisper_task=whisper_task,
                        whisper_language=whisper_language,
                        align_srt=srt_path,
                    )
                except Exception as exc:
                    error_container[0] = exc

            analyze_thread = threading.Thread(target=_analyze_with_timeout, name="VideoForgeAnalyze")
            analyze_thread.daemon = True
            analyze_thread.start()

            timeout = 600
            elapsed = 0
            while analyze_thread.is_alive() and elapsed < timeout:
                analyze_thread.join(timeout=5)
                elapsed += 5
                if elapsed % 15 == 0:
                    logger.info("[2/6] Still processing... (%d seconds elapsed)", elapsed)
                    self._update_status_safe(f"Step 2/6: Processing... ({elapsed}s elapsed)")
                    self._update_progress_safe(min(90, 16 + int(elapsed / timeout * 70)))

            if analyze_thread.is_alive():
                logger.error("[2/6] TIMEOUT after %d seconds!", timeout)
                raise TimeoutError(
                    f"Analyze timed out after {timeout}s. Try CPU mode or check CUDA/CT2 settings."
                )

            if error_container[0]:
                raise error_container[0]

            result = result_container[0] or {}

            logger.info("[3/6] Finalizing")
            self._update_progress_safe(95)
            self._update_status_safe("Step 3/6: Finalizing...")

            clips = self.bridge.resolve_api.get_selected_clips()
            main_path = clips[0].GetClipProperty("File Path") if clips else None
            result["main_path"] = main_path

            logger.info("[4/6] Analyze complete")
            self._update_progress_safe(100)
            self._update_status_safe("Step 4/6: Complete")
            return result

        def _done(result):
            self.project_db = result.get("project_db")
            self.main_video_path = result.get("main_path")
            self.analyze_status.setText("Status: Analyzed")
            self.analyze_status.setStyleSheet(f"color: {COLORS['success']}; margin-top: 6px;")
            warning = result.get("warning")
            if warning:
                self._set_status(f"Warning: {warning}")
            else:
                self._set_status("Analysis complete")

        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self._set_status("Analyzing... Check log for progress")
        self._run_worker(_analyze, on_done=_done)

    def _on_browse_library(self) -> None:
        default_dir = (
            self.saved_broll_dir
            if self.saved_broll_dir and Path(self.saved_broll_dir).exists()
            else ""
        )
        folder = QFileDialog.getExistingDirectory(self, "Select B-roll Folder", default_dir)
        if folder:
            Config.set("broll_dir", folder)
            self.saved_broll_dir = folder
            self.library_path_edit.setText(str(Path(folder) / "library.db"))

    def _on_scan_library(self) -> None:
        default_dir = (
            self.saved_broll_dir
            if self.saved_broll_dir and Path(self.saved_broll_dir).exists()
            else ""
        )
        folder = QFileDialog.getExistingDirectory(self, "Select B-roll Folder", default_dir)
        if not folder:
            return
        Config.set("broll_dir", folder)
        self.saved_broll_dir = folder

        def _scan():
            library_path = self.library_path_edit.text().strip()
            if not library_path:
                library_path = str(
                    self._resolve_repo_root() / "VideoForge" / "data" / "cache" / "library.db"
                )
                self.library_path_edit.setText(library_path)
            db = LibraryDB(
                library_path,
                schema_path=str(self._resolve_repo_root() / "VideoForge" / "config" / "schema.sql"),
            )
            count = scan_library(folder, db)
            return count

        def _done(result):
            self.library_status.setText(f"Status: Indexed {result} clips")
            self.library_status.setStyleSheet(f"color: {COLORS['success']}; margin-top: 6px;")
            self._set_status("Library scan complete")
            self.bridge.save_library_path(self.library_path_edit.text().strip())

        self._set_status("Scanning library...")
        self._run_worker(_scan, on_done=_done)

    def _on_browse_srt(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select SRT File", "", "SRT Files (*.srt)")
        if path:
            self.srt_path_edit.setText(path)

    def _on_open_log(self) -> None:
        log_path = self._get_log_path()
        if log_path and log_path.exists():
            os.startfile(str(log_path))
            self._set_status("Opened log file.")
            return
        if log_path and log_path.parent.exists():
            os.startfile(str(log_path.parent))
            self._set_status("Log file not found yet. Opened log folder.")
            return
        self._set_status("Log path unavailable.")

    def _on_match_clicked(self) -> None:
        if not self.project_db:
            self._set_status("Analyze a clip first.")
            return
        library_path = self.library_path_edit.text().strip()
        if not library_path:
            default_dir = (
                self.saved_broll_dir
                if self.saved_broll_dir and Path(self.saved_broll_dir).exists()
                else ""
            )
            folder = QFileDialog.getExistingDirectory(self, "Select B-roll Folder", default_dir)
            if not folder:
                self._set_status("Set a library DB path first.")
                return
            Config.set("broll_dir", folder)
            self.saved_broll_dir = folder
            library_path = str(Path(folder) / "library.db")
            self.library_path_edit.setText(library_path)
        threshold = self.threshold_slider.value() / 100.0
        if not 0.0 <= threshold <= 1.0:
            self._set_status(f"Invalid threshold: {threshold}")
            return
        self.settings["broll"]["matching_threshold"] = threshold

        def _match():
            return self.bridge.match_broll(self.project_db, library_path)

        def _done(result):
            count = result.get("count", 0)
            warning = result.get("warning")
            if warning:
                self._set_status(f"Warning: {warning}")
            else:
                self._set_status(f"Matched {count} clips")

        self._set_status("Matching... (searching library)")
        self._run_worker(_match, on_done=_done)

    def _on_apply_clicked(self) -> None:
        if not self.project_db or not self.main_video_path:
            self._set_status("Analyze a clip first.")
            return
        if not Path(self.main_video_path).exists():
            self._set_status(f"Main video not found: {Path(self.main_video_path).name}")
            return
        if not Path(self.project_db).exists():
            self._set_status("Project database missing. Re-run analysis.")
            return

        def _apply():
            self.bridge.apply_to_timeline(self.project_db, self.main_video_path)
            return True

        def _done(_result):
            self._set_status("Applied to timeline")

        self._set_status("Applying to timeline...")
        self._run_worker(_apply, on_done=_done)

    def _on_generate_sidecars(self) -> None:
        default_dir = (
            self.saved_broll_dir
            if self.saved_broll_dir and Path(self.saved_broll_dir).exists()
            else ""
        )
        folder = QFileDialog.getExistingDirectory(self, "Select B-roll Folder", default_dir)
        if not folder:
            return

        def _generate():
            return generate_scene_sidecars(Path(folder), overwrite=True)

        def _done(result):
            self._set_status(f"Generated {result} sidecars")

        self._set_status("Generating scene sidecars...")
        self._run_worker(_generate, on_done=_done)

    def _on_generate_comfyui_sidecars(self) -> None:
        default_dir = (
            self.saved_broll_dir
            if self.saved_broll_dir and Path(self.saved_broll_dir).exists()
            else ""
        )
        folder = QFileDialog.getExistingDirectory(self, "Select B-roll Folder", default_dir)
        if not folder:
            return

        def _generate():
            return generate_comfyui_sidecars(Path(folder), overwrite=True, sequence=True)

        def _done(result):
            self._set_status(f"Generated {result} ComfyUI sidecars")

        self._set_status("Generating ComfyUI sidecars...")
        self._run_worker(_generate, on_done=_done)

    def _update_last_log(self) -> None:
        log_path = self._get_log_path()
        if not log_path or not log_path.exists():
            return
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
                lines = handle.readlines()
            if not lines:
                return
            last_line = lines[-1].strip()
            if "] " in last_line:
                message = last_line.split("] ", 1)[-1]
            else:
                message = last_line
            self.last_log_label.setText(f"Last log: {message[:120]}")
        except Exception:
            return

    def _restore_library_path(self) -> None:
        try:
            saved = self.bridge.get_saved_library_path()
        except Exception:
            saved = None
        if saved:
            self.library_path_edit.setText(str(saved))
            return
        if self.saved_broll_dir:
            self.library_path_edit.setText(str(Path(self.saved_broll_dir) / "library.db"))
