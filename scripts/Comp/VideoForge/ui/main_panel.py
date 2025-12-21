from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml

from VideoForge.broll.db import LibraryDB
from VideoForge.broll.indexer import scan_library
from VideoForge.plugin.resolve_bridge import ResolveBridge

try:
    from PySide6.QtCore import QObject, QThread, Signal, Qt
    from PySide6.QtWidgets import (
        QApplication,
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
        from PySide2.QtCore import QObject, QThread, Signal, Qt
        from PySide2.QtWidgets import (
            QApplication,
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
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result, None)
        except Exception as exc:
            self.finished.emit(None, exc)


DEFAULT_SETTINGS = {
    "silence": {
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
    },
    "query": {"top_n_tokens": 6},
    "vision": {"max_frames": 12, "base_frames": 6},
    "resolve": {"timeline_name_prefix": "VideoForge_"},
}


class VideoForgePanel(QWidget):
    """Modern styled panel for VideoForge plugin."""

    def __init__(self) -> None:
        super().__init__()
        self._startup_warning: Optional[str] = None
        self.settings = self._load_settings()
        self.bridge = ResolveBridge(self.settings)
        self.project_db: Optional[str] = None
        self.main_video_path: Optional[str] = None
        self._threads: list[QThread] = []
        self._init_ui()
        if self._startup_warning:
            self._set_status(self._startup_warning)
        self._load_saved_library_path()

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

        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(subtitle)
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
        self.analyze_status = QLabel("○ Not analyzed")
        self.analyze_status.setObjectName("StatusLabel")
        card_layout.addWidget(self.analyze_status)

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

        self.browse_btn = QPushButton("📁")
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
        self.library_status = QLabel("○ Library not loaded")
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

        self.threshold_label = QLabel("65%")
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
        self.threshold_slider.setValue(65)
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

    def _set_busy(self, busy: bool) -> None:
        self.progress.setVisible(busy)
        self.analyze_btn.setEnabled(not busy)
        self.scan_library_btn.setEnabled(not busy)
        self.match_btn.setEnabled(not busy)
        self.apply_btn.setEnabled(not busy)

    def _run_worker(self, func, *args, on_done=None) -> None:
        self._set_busy(True)
        thread = QThread(self)
        worker = Worker(func, *args)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(lambda result, err: self._on_worker_done(thread, result, err, on_done))
        thread.start()
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

    # --- Event Handlers ---

    def _on_analyze_clicked(self) -> None:
        def _analyze():
            project_db = self.bridge.analyze_selected_clip()
            clips = self.bridge.resolve_api.get_selected_clips()
            main_path = clips[0].GetClipProperty("File Path") if clips else None
            return project_db, main_path

        def _done(result):
            self.project_db, self.main_video_path = result
            self.analyze_status.setText("✓ Analyzed")
            self.analyze_status.setStyleSheet(f"color: {COLORS['success']}; margin-top: 6px;")
            self._set_status("Analysis complete")

        self._set_status("Analyzing... (silence + transcription)")
        self._run_worker(_analyze, on_done=_done)

    def _on_browse_library(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Library DB", "", "DB Files (*.db)")
        if path:
            self.library_path_edit.setText(path)

    def _on_scan_library(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select B-roll Folder")
        if not folder:
            return

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
            self.library_status.setText(f"✓ Indexed {result} clips")
            self.library_status.setStyleSheet(f"color: {COLORS['success']}; margin-top: 6px;")
            self._set_status("Library scan complete")
            self.bridge.save_library_path(self.library_path_edit.text().strip())

        self._set_status("Scanning library...")
        self._run_worker(_scan, on_done=_done)

    def _on_match_clicked(self) -> None:
        if not self.project_db:
            self._set_status("Analyze a clip first.")
            return
        library_path = self.library_path_edit.text().strip()
        if not library_path:
            self._set_status("Set a library DB path first.")
            return
        threshold = self.threshold_slider.value() / 100.0
        self.settings["broll"]["matching_threshold"] = threshold

        def _match():
            return self.bridge.match_broll(self.project_db, library_path)

        def _done(result):
            self._set_status(f"Matched {result} clips")

        self._set_status("Matching... (searching library)")
        self._run_worker(_match, on_done=_done)

    def _on_apply_clicked(self) -> None:
        if not self.project_db or not self.main_video_path:
            self._set_status("Analyze a clip first.")
            return

        def _apply():
            self.bridge.apply_to_timeline(self.project_db, self.main_video_path)
            return True

        def _done(_result):
            self._set_status("Applied to timeline ✓")

        self._set_status("Applying to timeline...")
        self._run_worker(_apply, on_done=_done)

    def _load_saved_library_path(self) -> None:
        try:
            saved = self.bridge.get_saved_library_path()
        except Exception:
            saved = None
        if saved:
            self.library_path_edit.setText(saved)
