from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import threading
from pathlib import Path
from typing import Optional

import yaml

from VideoForge.broll.db import LibraryDB
from VideoForge.broll.metadata import coerce_metadata_dict
from VideoForge.broll.indexer import LibraryIndexer, scan_library
from VideoForge.broll.sidecar_tools import generate_comfyui_sidecars, generate_scene_sidecars
from VideoForge.config.config_manager import Config
from VideoForge.core.segment_store import SegmentStore
from VideoForge.core.project_cleanup import cleanup_project_files
from VideoForge.plugin.resolve_bridge import ResolveBridge
from VideoForge.core.silence_processor import (
    process_silence,
    render_silence_removed_with_reorder,
    build_ordered_segments_for_draft,
)
from VideoForge.adapters.audio_adapter import render_silence_removed
from VideoForge.errors import VideoForgeError
from VideoForge.ai.llm_hooks import is_llm_enabled, write_llm_env
from VideoForge.ui.qt_compat import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QObject,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QTabWidget,
    QThread,
    Qt,
    QTimer,
    QVBoxLayout,
    QWidget,
    Signal,
)
from VideoForge.ui.sections.analyze_section import build_analyze_section
from VideoForge.ui.sections.library_section import build_library_section
from VideoForge.ui.sections.match_section import build_match_section
from VideoForge.ui.sections.misc_section import build_misc_section
from VideoForge.ui.sections.settings_section import build_settings_section


# --- Color Palette (Parchment & Mist) ---
COLORS = {
    "bg_window": "#e9e5de",
    "bg_card": "#fdfbf7",
    "text_main": "#1a1a1a",
    "text_dim": "rgba(26, 26, 26, 0.6)",
    "accent": "#d4cfc3",
    "accent_hover": "#c9c4b8",
    "button_bg": "rgba(255, 255, 255, 0.5)",
    "button_hover": "rgba(255, 255, 255, 0.8)",
    "border": "rgba(0, 0, 0, 0.05)",
    "highlight": "rgba(0, 0, 0, 0.02)",
    "success": "#4caf50",
    "error": "#d32f2f",
}

VERSION = "v1.8.0"

# --- Stylesheet ---
STYLESHEET = f"""
    QWidget {{
        background-color: {COLORS['bg_window']};
        font-family: 'Inter', 'Segoe UI', sans-serif;
        font-size: 13px;
        color: {COLORS['text_main']};
    }}

    /* Section Cards */
    QFrame#SectionCard {{
        background-color: {COLORS['bg_card']};
        border-radius: 4px;
        border: 1px solid {COLORS['border']};
        padding: 16px;
    }}

    /* Section Titles */
    QLabel#SectionTitle {{
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        font-weight: bold;
        color: {COLORS['text_main']};
        letter-spacing: 0.15em;
        text-transform: uppercase;
        padding-bottom: 8px;
        opacity: 0.8;
    }}

    /* Description Labels */
    QLabel#Description {{
        color: {COLORS['text_dim']};
        margin-bottom: 8px;
        font-style: italic;
    }}

    /* Buttons */
    QPushButton {{
        background-color: {COLORS['button_bg']};
        border: 1px solid {COLORS['border']};
        border-radius: 2px;
        padding: 6px 12px;
        color: {COLORS['text_main']};
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        text-transform: lowercase;
    }}
    QPushButton:hover {{
        background-color: {COLORS['button_hover']};
        border: 1px solid {COLORS['accent']};
    }}
    QPushButton#PrimaryButton {{
        background-color: {COLORS['accent']};
        border: none;
    }}
    QPushButton#PrimaryButton:hover {{
        background-color: {COLORS['accent_hover']};
    }}

    /* Text Inputs & Combos */
    QLineEdit, QComboBox {{
        background-color: white;
        border: 1px solid {COLORS['border']};
        border-radius: 2px;
        padding: 4px 8px;
        color: {COLORS['text_main']};
    }}
    QLineEdit:focus, QComboBox:focus {{
        border: 1px solid {COLORS['accent']};
    }}

    /* Tabs */
    QTabWidget::pane {{
        border: none;
        background: transparent;
    }}
    QTabBar::tab {{
        background: transparent;
        padding: 8px 16px;
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: {COLORS['text_dim']};
    }}
    QTabBar::tab:selected {{
        color: {COLORS['text_main']};
        border-bottom: 2px solid {COLORS['text_main']};
    }}

    /* Scroll Area */
    QScrollArea {{
        border: none;
        background: transparent;
    }}
    
    /* Sliders */
    QSlider::groove:horizontal {{
        border: none;
        height: 2px;
        background: {COLORS['border']};
    }}
    QSlider::handle:horizontal {{
        background: {COLORS['text_main']};
        width: 10px;
        height: 10px;
        margin: -4px 0;
        border-radius: 5px;
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
        "render_min_duration": 0.6,
        "tail_min_duration": 0.8,
        "tail_ratio": 0.2,
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
        "jlcut_mode": "off",
        "jcut_offset": 0.0,
        "lcut_offset": 0.0,
        "jlcut_overlap_frames": 15,
        "jlcut_overlap_jitter": 0,
    },
    "whisper": {"task": "transcribe", "language": "auto"},
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
        logging.getLogger("VideoForge.ui").info("UI init start (%s)", VERSION)
        self._startup_warning: Optional[str] = None
        self.colors = COLORS
        self.settings = self._load_settings()
        self._apply_broll_overrides()
        whisper_silence_override = Config.get("whisper_silence_enabled")
        if whisper_silence_override is not None:
            self.settings["silence"]["enable_removal"] = bool(whisper_silence_override)
        self._apply_silence_override("render_min_duration")
        self._apply_silence_override("tail_min_duration")
        self._apply_silence_override("tail_ratio")
        self.bridge = ResolveBridge(self.settings)
        self.saved_broll_dir = Config.get("broll_dir")
        self.project_db: Optional[str] = None
        self.main_video_path: Optional[str] = None
        self._threads: list[QThread] = []
        self._workers: list[Worker] = []
        self.transcript_panel = None
        self._threads_lock = threading.Lock()
        self.status_signal.connect(self._set_status)
        self.progress_signal.connect(self._set_progress_value)
        self._init_ui()
        self._start_resolve_watchdog()
        if self._startup_warning:
            self._set_status(self._startup_warning)
        try:
            preset = Config.get("silence_preset")
            if preset:
                self._on_silence_preset_changed(str(preset))
        except Exception:
            pass
        self._restore_library_path()
        self._restore_project_state()
        logging.getLogger("VideoForge.ui").info("UI init done (%s)", VERSION)

    def _get_project_key(self) -> Optional[str]:
        try:
            return self.bridge.resolve_api.get_project_key()
        except Exception as exc:
            logging.getLogger("VideoForge.ui").warning(
                "Failed to read project key: %s", exc
            )
            return None

    def _save_project_state(self) -> None:
        project_key = self._get_project_key()
        if not project_key or not self.project_db:
            return
        project_map = Config.get("project_db_map", {})
        if not isinstance(project_map, dict):
            project_map = {}
        project_map[project_key] = {
            "project_db": self.project_db,
            "main_video_path": self.main_video_path,
        }
        Config.set("project_db_map", project_map)

    def _restore_project_state(self) -> None:
        project_key = self._get_project_key()
        if not project_key:
            return
        project_map = Config.get("project_db_map", {})
        if not isinstance(project_map, dict):
            return
        payload = project_map.get(project_key)
        if not isinstance(payload, dict):
            return
        project_db = payload.get("project_db")
        main_path = payload.get("main_video_path")
        if project_db and Path(str(project_db)).exists():
            self.project_db = str(project_db)
            self.main_video_path = str(main_path) if main_path else None
            self.analyze_status.setText("Status: Loaded")
            self.analyze_status.setStyleSheet(f"color: {COLORS['success']}; margin-top: 6px;")
            self.edit_transcript_btn.setEnabled(True)
            logging.getLogger("VideoForge.ui").info(
                "Restored project DB for %s", project_key
            )
    def _init_ui(self) -> None:
        """Initialize the modernized UI."""
        self.setStyleSheet(STYLESHEET)
        self.setWindowTitle("VideoForge - Auto B-roll")
        self.resize(1200, 720)
        self.setMinimumSize(800, 500)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        # --- Header (Always Visible) ---
        self._setup_header(layout)

        # --- Tabs ---
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Tab 1: Analyze
        analyze_scroll = QScrollArea()
        analyze_scroll.setWidgetResizable(True)
        analyze_scroll.setFrameShape(QFrame.NoFrame)
        analyze_tab = QWidget()
        analyze_scroll.setWidget(analyze_tab)
        
        analyze_layout = QHBoxLayout(analyze_tab)
        analyze_layout.setContentsMargins(8, 8, 8, 8)
        analyze_layout.setSpacing(12)
        
        # Left side: Analyze Main Clip
        analyze_left_col = QVBoxLayout()
        build_analyze_section(self, analyze_left_col)
        analyze_left_col.addStretch()
        
        # Right side: B-roll Library
        analyze_right_col = QVBoxLayout()
        build_library_section(self, analyze_right_col)
        analyze_right_col.addStretch()
        
        analyze_layout.addLayout(analyze_left_col, 1)
        analyze_layout.addLayout(analyze_right_col, 1)
        
        self.tabs.addTab(analyze_scroll, "📝 Analyze")

        # Tab 2: Match
        match_scroll = QScrollArea()
        match_scroll.setWidgetResizable(True)
        match_scroll.setFrameShape(QFrame.NoFrame)
        match_tab = QWidget()
        match_scroll.setWidget(match_tab)
        
        match_layout = QVBoxLayout(match_tab)
        match_layout.setContentsMargins(8, 8, 8, 8)
        build_match_section(self, match_layout)
        match_layout.addStretch()
        self.tabs.addTab(match_scroll, "🎬 Match")

        # Tab 3: Settings
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QFrame.NoFrame)
        settings_tab = QWidget()
        settings_scroll.setWidget(settings_tab)
        
        settings_layout = QVBoxLayout(settings_tab)
        settings_layout.setContentsMargins(8, 8, 8, 8)
        build_settings_section(self, settings_layout)
        settings_layout.addStretch()
        self.tabs.addTab(settings_scroll, "⚙️ Settings")

        # Tab 4: Misc
        misc_scroll = QScrollArea()
        misc_scroll.setWidgetResizable(True)
        misc_scroll.setFrameShape(QFrame.NoFrame)
        misc_tab = QWidget()
        misc_scroll.setWidget(misc_tab)

        misc_layout = QVBoxLayout(misc_tab)
        misc_layout.setContentsMargins(8, 8, 8, 8)
        build_misc_section(self, misc_layout)
        misc_layout.addStretch()
        self.tabs.addTab(misc_scroll, "🧰 Misc")

        # --- Footer (Always Visible) ---
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
        candidate = script_root / "VideoForge"
        if candidate.exists():
            return candidate
        env_root = os.environ.get("VIDEOFORGE_ROOT")
        if env_root:
            env_path = Path(env_root)
            if env_path.exists():
                return env_path
        root = Path(__file__).resolve().parents[3]
        if not root.exists():
            raise RuntimeError("Failed to resolve VideoForge root path.")
        return root

    @staticmethod
    def _ensure_unique_path(path: Path) -> Path:
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        for idx in range(2, 1000):
            candidate = parent / f"{stem}_{idx}{suffix}"
            if not candidate.exists():
                return candidate
        return path

    def _set_status(self, message: str) -> None:
        self.status.setText(message)

    def _apply_silence_override(self, key: str) -> None:
        override = Config.get(f"silence_{key}")
        if override is None:
            return
        try:
            value = float(override)
        except (TypeError, ValueError):
            return
        self.settings["silence"][key] = value

    def _apply_broll_overrides(self) -> None:
        broll = self.settings.setdefault("broll", {})
        mode = Config.get("jlcut_mode")
        if isinstance(mode, str) and mode.strip():
            broll["jlcut_mode"] = mode.strip().lower()
        for key in ("jcut_offset", "lcut_offset", "jlcut_overlap_frames", "jlcut_overlap_jitter"):
            override = Config.get(key)
            if override is None:
                continue
            try:
                if key == "jlcut_overlap_frames":
                    broll[key] = int(float(override))
                elif key == "jlcut_overlap_jitter":
                    broll[key] = int(float(override))
                else:
                    broll[key] = float(override)
            except (TypeError, ValueError):
                continue

    def _get_broll_output_root(self) -> Path:
        default_root = self._resolve_repo_root() / "generated_broll"
        raw = (
            Config.get("broll_gen_output_dir")
            or Config.get("uvm_output_dir")
            or str(default_root)
        )
        return Path(str(raw))

    def _get_broll_project_name(self) -> str:
        if self.project_db:
            return Path(self.project_db).stem
        return "project"

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
        for attr in ("scan_library_btn", "scan_global_library_btn", "scan_local_library_btn"):
            button = getattr(self, attr, None)
            if button:
                button.setEnabled(not busy)
        self.generate_broll_btn.setEnabled(not busy)
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
        worker.finished.connect(
            lambda result, err, t=thread, w=worker: self._on_worker_done(t, w, result, err, on_done),
            Qt.QueuedConnection,
        )
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.start()
        with self._threads_lock:
            self._threads.append(thread)
            self._workers.append(worker)

    def _on_worker_done(self, thread: QThread, worker: Worker, result, err, on_done) -> None:
        self._set_busy(False)
        if err:
            if isinstance(err, VideoForgeError):
                logging.getLogger("VideoForge.ui").error(
                    "VideoForge error: %s", err, extra=getattr(err, "details", None)
                )
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
            if worker in self._workers:
                self._workers.remove(worker)

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

    def _on_task_changed(self, value: str) -> None:
        self.settings.setdefault("whisper", {})["task"] = value

    def _on_transcript_chars_changed(self, value: str) -> None:
        try:
            Config.set("subtitle_max_chars", int(str(value).strip()))
        except Exception:
            return

    def _on_render_min_duration_changed(self, value: str) -> None:
        try:
            duration = float(str(value).strip())
        except ValueError:
            return
        if duration <= 0:
            return
        self.settings["silence"]["render_min_duration"] = duration
        Config.set("silence_render_min_duration", duration)
        try:
            self.silence_preset_combo.setCurrentText("Custom")
        except Exception:
            pass

    def _on_tail_min_duration_changed(self, value: str) -> None:
        try:
            duration = float(str(value).strip())
        except ValueError:
            return
        if duration <= 0:
            return
        self.settings["silence"]["tail_min_duration"] = duration
        Config.set("silence_tail_min_duration", duration)
        try:
            self.silence_preset_combo.setCurrentText("Custom")
        except Exception:
            pass

    def _on_tail_ratio_changed(self, value: str) -> None:
        try:
            ratio = float(str(value).strip())
        except ValueError:
            return
        if ratio <= 0 or ratio >= 1:
            return
        self.settings["silence"]["tail_ratio"] = ratio
        Config.set("silence_tail_ratio", ratio)
        try:
            self.silence_preset_combo.setCurrentText("Custom")
        except Exception:
            pass

    def _on_render_mode_changed(self, value: str) -> None:
        mode = str(value).strip()
        if mode not in {"Silence Removal", "J/L-cut"}:
            mode = "Silence Removal"
        Config.set("render_mode", mode)

    def _on_llm_provider_changed(self, value: str) -> None:
        provider = str(value).strip().lower()
        if provider == "disabled":
            Config.set("llm_provider", "disabled")
            write_llm_env(provider="")
        else:
            Config.set("llm_provider", provider)
            write_llm_env(provider=provider)
        enabled = provider != "disabled"
        if hasattr(self, "llm_model_edit"):
            self.llm_model_edit.setEnabled(enabled)
        if hasattr(self, "llm_key_edit"):
            self.llm_key_edit.setEnabled(enabled)
        self._update_llm_buttons()

    def _on_llm_model_changed(self, value: str) -> None:
        model = str(value).strip()
        if model.startswith("models/"):
            model = model[len("models/"):]
            try:
                self.llm_model_edit.setText(model)
            except Exception:
                pass
        if model:
            Config.set("llm_model", model)
            write_llm_env(model=model)
        self._update_llm_buttons()

    def _on_llm_key_changed(self, value: str) -> None:
        Config.set("llm_api_key", str(value))
        write_llm_env(api_key=str(value))
        self._update_llm_buttons()

    def _on_pexels_key_changed(self, value: str) -> None:
        Config.set("pexels_api_key", str(value).strip())

    def _on_unsplash_key_changed(self, value: str) -> None:
        Config.set("unsplash_api_key", str(value).strip())

    def _on_comfyui_url_changed(self, value: str) -> None:
        Config.set("comfyui_url", str(value).strip())

    def _on_comfyui_output_changed(self, value: str) -> None:
        """ComfyUI output directory path (optional)."""
        Config.set("comfyui_output_dir", str(value).strip())

    def _on_comfyui_output_browse(self) -> None:
        """Select ComfyUI output directory (optional)."""
        start_dir = str(Config.get("comfyui_output_dir") or "C:/ComfyUI/ComfyUI/output")
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select ComfyUI Output Directory",
            start_dir,
        )
        if folder:
            try:
                self.comfyui_output_edit.setText(folder)
            except Exception:
                pass
            Config.set("comfyui_output_dir", folder)

    def _on_comfyui_resolution_changed(self, value: str) -> None:
        """ComfyUI resolution selection."""
        reverse_map = {
            "1920x1024 (default)": "1920x1024",
            "1920x1080 (16:9)": "1920x1080",
            "1024x1024 (1:1)": "1024x1024",
            "1024x1824 (9:16)": "1024x1824",
        }
        Config.set("comfyui_resolution", reverse_map.get(value, "1920x1024"))

    def _on_comfyui_seed_changed(self, value: str) -> None:
        """Optional ComfyUI seed (blank = random each run)."""
        text = str(value).strip()
        if not text:
            Config.set("comfyui_seed", "")
            return
        if text.isdigit():
            Config.set("comfyui_seed", text)

    def _on_comfyui_run_script_changed(self, value: str) -> None:
        """ComfyUI run script path (optional)."""
        Config.set("comfyui_run_script", str(value).strip())

    def _on_comfyui_run_script_browse(self) -> None:
        """Select ComfyUI run script (optional)."""
        start_dir = str(Config.get("comfyui_run_script") or "C:/ComfyUI/ComfyUI/run_comfyui.bat")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select ComfyUI Run Script",
            start_dir,
            "Batch Files (*.bat);;All Files (*)",
        )
        if path:
            try:
                self.comfyui_run_script_edit.setText(path)
            except Exception:
                pass
            Config.set("comfyui_run_script", path)

    def _on_comfyui_autostart_changed(self, state: int) -> None:
        """Toggle ComfyUI autostart."""
        Config.set("comfyui_autostart", bool(state))

    def _on_scene_detection_changed(self, state: int) -> None:
        """Toggle OpenCV scene detection for sampling."""
        Config.set("use_scene_detection", bool(state))

    def _on_scene_threshold_changed(self, value: int) -> None:
        """Adjust scene detection sensitivity."""
        Config.set("scene_detection_threshold", int(value))
        try:
            self.scene_threshold_label.setText(f"Sensitivity: {int(value)}")
        except Exception:
            pass

    def _on_quality_check_changed(self, state: int) -> None:
        """Toggle stock/AI quality check."""
        Config.set("enable_quality_check", bool(state))

    def _on_min_quality_changed(self, value: int) -> None:
        """Adjust minimum quality score threshold."""
        Config.set("min_quality_score", float(value) / 100.0)
        try:
            self.min_quality_label.setText(f"Min Quality: {int(value)}%")
        except Exception:
            pass

    def _on_sam3_tagging_changed(self, state: int) -> None:
        """Toggle SAM3 auto-tagging."""
        enabled = bool(state)
        Config.set("use_sam3_tagging", enabled)
        logging.getLogger("VideoForge.ui").info(
            "SAM3 Auto-Tagging: %s", "enabled" if enabled else "disabled"
        )

    def _on_sam3_model_changed(self, model_size: str) -> None:
        """Set SAM3 model size."""
        value = str(model_size).strip()
        if value:
            Config.set("sam3_model_size", value)
            logging.getLogger("VideoForge.ui").info("SAM3 Model Size: %s", value)

    def _on_sam3_prompt_changed(self, value: str) -> None:
        Config.set("sam3_prompt", value)

    def _on_sam3_input_changed(self, value: str) -> None:
        Config.set("sam3_input_path", value)

    def _on_sam3_output_changed(self, value: str) -> None:
        Config.set("sam3_output_dir", value)

    def _on_sam3_input_browse(self) -> None:
        start_dir = str(Config.get("sam3_input_path") or "")
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            start_dir,
            "Video Files (*.mp4 *.mov *.mkv *.avi);;All Files (*)",
        )
        if not file_path:
            return
        self.sam3_input_edit.setText(file_path)
        Config.set("sam3_input_path", file_path)

    def _on_sam3_output_browse(self) -> None:
        start_dir = str(Config.get("sam3_output_dir") or "")
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            start_dir,
        )
        if not folder:
            return
        self.sam3_output_edit.setText(folder)
        Config.set("sam3_output_dir", folder)

    def _on_sam3_run_clicked(self) -> None:
        prompt = (self.sam3_prompt_edit.text() or "").strip()
        if not prompt:
            self._set_status("Enter a SAM3 prompt first.")
            return
        source_path = (self.sam3_input_edit.text() or "").strip()
        if not source_path:
            self._set_status("Select a video file.")
            return
        source = Path(source_path)
        if not source.exists():
            self._set_status(f"Input file not found: {source.name}")
            return

        output_dir = (self.sam3_output_edit.text() or "").strip()
        output_root = Path(output_dir) if output_dir else source.parent
        output_root.mkdir(parents=True, exist_ok=True)
        output_path = output_root / f"{source.stem}_sam3.json"

        try:
            model_size = str(Config.get("sam3_model_size") or "large")
        except Exception:
            model_size = "large"

        def _run():
            from VideoForge.adapters.sam3_subprocess import SAM3Subprocess

            adapter = SAM3Subprocess()
            result = adapter.segment_video(
                source,
                prompt,
                model_size=model_size,
                max_frames=300,
            )
            if not result:
                return None
            output_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return str(output_path)

        def _done(result):
            if result:
                self._set_status(f"SAM3 saved: {Path(result).name}")
            else:
                self._set_status("SAM3 failed. Check prompt or WSL logs.")

        self._set_status("Running SAM3 segmentation...")
        self._run_worker(_run, on_done=_done)

    def _on_sam_audio_model_changed(self, model_size: str) -> None:
        """Set SAM Audio model size."""
        value = str(model_size).strip()
        if value:
            Config.set("sam_audio_model_size", value)
            logging.getLogger("VideoForge.ui").info("SAM Audio Model Size: %s", value)

    def _on_sam_audio_prompt_changed(self, value: str) -> None:
        Config.set("sam_audio_prompt", value)

    def _on_sam_audio_input_changed(self, value: str) -> None:
        Config.set("sam_audio_input_path", value)

    def _on_sam_audio_output_changed(self, value: str) -> None:
        Config.set("sam_audio_output_dir", value)

    def _on_sam_audio_input_browse(self) -> None:
        start_dir = str(Config.get("sam_audio_input_path") or "")
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio/Video File",
            start_dir,
            "Media Files (*.wav *.mp3 *.m4a *.flac *.aac *.mp4 *.mov *.mkv);;All Files (*)",
        )
        if not file_path:
            return
        self.sam_audio_input_edit.setText(file_path)
        Config.set("sam_audio_input_path", file_path)

    def _on_sam_audio_output_browse(self) -> None:
        start_dir = str(Config.get("sam_audio_output_dir") or "")
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            start_dir,
        )
        if not folder:
            return
        self.sam_audio_output_edit.setText(folder)
        Config.set("sam_audio_output_dir", folder)

    def _on_sam_audio_run_clicked(self) -> None:
        prompt = (self.sam_audio_prompt_edit.text() or "").strip()
        if not prompt:
            self._set_status("Enter an isolation prompt first.")
            return
        source_path = (self.sam_audio_input_edit.text() or "").strip()
        if not source_path:
            self._set_status("Select an audio/video file.")
            return
        source = Path(source_path)
        if not source.exists():
            self._set_status(f"Input file not found: {source.name}")
            return

        output_dir = (self.sam_audio_output_edit.text() or "").strip()
        output_path = None
        if output_dir:
            output_root = Path(output_dir)
            output_root.mkdir(parents=True, exist_ok=True)
            output_path = output_root / f"{source.stem}_isolated.wav"

        def _run():
            from VideoForge.adapters.sam_audio_adapter import SAMAudioAdapter

            adapter = SAMAudioAdapter()
            return adapter.separate_audio(source, prompt=prompt, output_path=output_path)

        def _done(result):
            if result:
                self._set_status(f"SAM Audio saved: {Path(result).name}")
            else:
                self._set_status("SAM Audio failed. Check prompt or WSL logs.")

        self._set_status("Running SAM Audio isolation...")
        self._run_worker(_run, on_done=_done)

    def _on_wsl_distro_changed(self, text: str) -> None:
        """Update WSL distro name."""
        value = str(text).strip()
        if not value:
            return
        Config.set("sam3_wsl_distro", value)
        Config.set("sam_audio_wsl_distro", value)

    def _on_wsl_python_changed(self, text: str) -> None:
        """Update WSL python command."""
        value = str(text).strip()
        if not value:
            return
        Config.set("sam3_wsl_python", value)
        Config.set("sam_audio_wsl_python", value)

    def _on_wsl_venv_changed(self, text: str) -> None:
        """Update WSL virtualenv path."""
        value = str(text).strip()
        Config.set("sam3_wsl_venv", value)
        Config.set("sam_audio_wsl_venv", value)

    def _on_llm_instructions_mode_changed(self, value: str) -> None:
        mode = str(value).strip().lower()
        if mode not in {"shortform", "longform"}:
            return
        Config.set("llm_instructions_mode", mode)
        write_llm_env(instructions_mode=mode)
        self._update_llm_buttons()

    def _on_engine_changed(self, value: str) -> None:
        Config.set("transcription_engine", value)
        self._update_transcript_option_visibility(value)

    def _on_silence_preset_changed(self, value: str) -> None:
        preset = str(value).strip()
        presets = {
            "Balanced": {"render_min_duration": 0.6, "tail_min_duration": 0.8, "tail_ratio": 0.2},
            "Aggressive": {"render_min_duration": 0.8, "tail_min_duration": 1.0, "tail_ratio": 0.25},
            "Conservative": {"render_min_duration": 0.4, "tail_min_duration": 0.6, "tail_ratio": 0.15},
        }
        if preset not in presets and preset != "Custom":
            return
        Config.set("silence_preset", preset)
        if preset == "Custom":
            return
        values = presets[preset]
        self.settings["silence"].update(values)
        Config.set("silence_render_min_duration", values["render_min_duration"])
        Config.set("silence_tail_min_duration", values["tail_min_duration"])
        Config.set("silence_tail_ratio", values["tail_ratio"])
        try:
            self.render_min_duration_edit.setText(str(values["render_min_duration"]))
            self.tail_min_duration_edit.setText(str(values["tail_min_duration"]))
            self.tail_ratio_edit.setText(str(values["tail_ratio"]))
        except Exception:
            pass

    def _update_transcript_option_visibility(self, engine: str) -> None:
        engine_key = str(engine).strip().lower()
        show = engine_key in {"resolve ai", "resolve_ai", "resolve"}
        if hasattr(self, "transcript_chars_label"):
            self.transcript_chars_label.setVisible(show)
        if hasattr(self, "transcript_chars_edit"):
            self.transcript_chars_edit.setVisible(show)
        self._update_llm_buttons()

    def _update_llm_buttons(self) -> None:
        try:
            enabled = is_llm_enabled()
        except Exception:
            enabled = False
        if hasattr(self, "llm_status_label"):
            status = "Active" if enabled else "Inactive"
            self.llm_status_label.setText(f"LLM Status: {status}")
        if getattr(self, "transcript_panel", None):
            try:
                self.transcript_panel.ai_reflow_btn.setEnabled(enabled)
                self.transcript_panel.ai_reorder_btn.setEnabled(enabled)
            except Exception:
                pass

    def _update_draft_render_emphasis(self) -> None:
        draft_active = False
        try:
            if self.project_db:
                store = SegmentStore(self.project_db)
                draft_active = bool(store.has_draft_order())
        except Exception:
            draft_active = False

        prev = bool(getattr(self, "_draft_render_emphasis", False))
        if prev == draft_active:
            return
        self._draft_render_emphasis = draft_active
        logging.getLogger("VideoForge.ui").info("Draft render emphasis: %s", draft_active)

        self._set_button_primary(getattr(self, "silence_export_btn", None), draft_active)
        self._set_button_primary(getattr(self, "silence_replace_btn", None), draft_active)

    @staticmethod
    def _set_button_primary(button: QPushButton | None, enabled: bool) -> None:
        if button is None:
            return
        target = "PrimaryButton" if enabled else ""
        if button.objectName() == target:
            return
        button.setObjectName(target)
        try:
            style = button.style()
            style.unpolish(button)
            style.polish(button)
            button.update()
        except Exception:
            return

    def _on_whisper_silence_toggled(self, state: int) -> None:
        enabled = state == Qt.Checked
        self.settings["silence"]["enable_removal"] = enabled
        Config.set("whisper_silence_enabled", enabled)

    def _on_detect_silence_clicked(self) -> None:
        clip = self.bridge.resolve_api.get_primary_clip()
        if not clip:
            self._set_status("Select a clip first.")
            return
        video_path = self.bridge._get_clip_path(clip)
        if not video_path:
            self._set_status("Selected clip has no file path.")
            return
        self._set_status("Detecting silence... (ffmpeg)")

        def _detect():
            segments = process_silence(
                video_path=video_path,
                threshold_db=self.settings["silence"]["threshold_db"],
                min_keep=self.settings["silence"]["min_keep_duration"],
                buffer_before=self.settings["silence"]["buffer_before"],
                buffer_after=self.settings["silence"]["buffer_after"],
            )
            logging.getLogger("VideoForge.ui").info(
                "Silence detect: %d keep segments for %s", len(segments), video_path
            )
            for idx, seg in enumerate(segments[:20], start=1):
                logging.getLogger("VideoForge.ui").info(
                    "Keep %d: %.2f - %.2f", idx, float(seg["t0"]), float(seg["t1"])
                )
            if len(segments) > 20:
                logging.getLogger("VideoForge.ui").info("... (%d more)", len(segments) - 20)
            return segments

        def _done(result):
            self._set_status(f"Silence segments: {len(result)} (see log)")

        self._run_worker(_detect, on_done=_done)

    def _on_apply_silence_to_timeline_clicked(self) -> None:
        clip = self.bridge.resolve_api.get_primary_clip()
        if not clip:
            self._set_status("Select a clip first.")
            return
        video_path = self.bridge._get_clip_path(clip)
        if not video_path:
            self._set_status("Selected clip has no file path.")
            return
        dry_run = self.silence_dry_run_checkbox.isChecked()
        self._set_status("Applying silence cuts... (debug)")

        def _apply():
            segments = process_silence(
                video_path=video_path,
                threshold_db=self.settings["silence"]["threshold_db"],
                min_keep=self.settings["silence"]["min_keep_duration"],
                buffer_before=self.settings["silence"]["buffer_before"],
                buffer_after=self.settings["silence"]["buffer_after"],
            )
            return self.bridge.resolve_api.debug_apply_silence_cuts(
                clip,
                segments,
                dry_run=dry_run,
            )

        def _done(result):
            if result:
                self._set_status("Silence cut debug complete (see log).")
            else:
                self._set_status("Silence cut not supported (see log).")

        self._run_worker(_apply, on_done=_done)

    def _on_render_silence_removed_clicked(self) -> None:
        clip = self.bridge.resolve_api.get_primary_clip()
        if not clip:
            self._set_status("Select a clip first.")
            return
        video_path = self.bridge._get_clip_path(clip)
        if not video_path:
            self._set_status("Selected clip has no file path.")
            return
        source_range = self.bridge.resolve_api.get_clip_source_range_seconds(clip)
        logging.getLogger("VideoForge.ui").info(
            "Silence render source range: %s",
            source_range,
        )
        self._set_status("Rendering silence-removed file...")

        def _render():
            jlcut_mode = self.settings.get("broll", {}).get("jlcut_mode", "off")
            jcut_offset = float(self.settings.get("broll", {}).get("jcut_offset", 0.0))
            lcut_offset = float(self.settings.get("broll", {}).get("lcut_offset", 0.0))
            render_mode = Config.get("render_mode", "Silence Removal")
            if render_mode not in {"Silence Removal", "J/L-cut"}:
                render_mode = "Silence Removal"
            overlap_frames = int(Config.get("jlcut_overlap_frames", 15) or 15)
            overlap_jitter = int(Config.get("jlcut_overlap_jitter", 0) or 0)
            logging.getLogger("VideoForge.ui").info(
                "Silence render J/L-cut settings: mode=%s j=%.3fs l=%.3fs",
                jlcut_mode,
                jcut_offset,
                lcut_offset,
            )
            logging.getLogger("VideoForge.ui").info(
                "Silence render mode: %s",
                render_mode,
            )
            logging.getLogger("VideoForge.ui").info(
                "J/L-cut overlap frames: %d",
                overlap_frames,
            )
            if overlap_jitter:
                logging.getLogger("VideoForge.ui").info(
                    "J/L-cut overlap jitter: +/- %d frames",
                    overlap_jitter,
                )
            keep_segments = process_silence(
                video_path=video_path,
                threshold_db=self.settings["silence"]["threshold_db"],
                min_keep=self.settings["silence"]["min_keep_duration"],
                buffer_before=self.settings["silence"]["buffer_before"],
                buffer_after=self.settings["silence"]["buffer_after"],
            )
            sentences = []
            draft_order = None
            store = None
            if self.project_db:
                store = SegmentStore(self.project_db)
                sentences = store.get_sentences()
                if store.has_draft_order():
                    draft_order = store.get_draft_order()
                    logging.getLogger("VideoForge.ui").info(
                        "Silence render using draft order (%d entries).",
                        len(draft_order),
                    )
            overlap_seed = None
            if store and overlap_jitter > 0:
                try:
                    existing_seed = store.get_artifact("render_overlap_seed")
                    existing_jitter = store.get_artifact("render_overlap_jitter")
                    if (
                        existing_seed is not None
                        and int(existing_jitter or 0) == overlap_jitter
                    ):
                        overlap_seed = int(existing_seed)
                    else:
                        overlap_seed = secrets.randbits(32)
                except Exception as exc:
                    logging.getLogger("VideoForge.ui").warning(
                        "Failed to resolve overlap seed: %s", exc
                    )

            def _save_render_artifacts(
                mode: str,
                segments: list[dict],
                overlap: int,
                source: tuple[float, float] | None,
                jitter: int = 0,
                seed: int | None = None,
            ) -> None:
                if not store:
                    return
                try:
                    store.save_artifact("render_mode", mode)
                    store.save_artifact("render_segments", segments)
                    store.save_artifact("render_overlap_frames", int(overlap))
                    store.save_artifact("render_overlap_jitter", int(jitter))
                    store.save_artifact("render_overlap_seed", seed)
                    store.save_artifact(
                        "render_source_range",
                        [float(source[0]), float(source[1])] if source else None,
                    )
                except Exception as exc:
                    logging.getLogger("VideoForge.ui").warning(
                        "Failed to save render artifacts: %s", exc
                    )
            base_path = Path(video_path)
            output_path = self._ensure_unique_path(
                base_path.with_name(f"{base_path.stem}_silence_removed{base_path.suffix}")
            )
            if render_mode == "J/L-cut":
                if not sentences:
                    logging.getLogger("VideoForge.ui").warning(
                        "J/L-cut requested but no transcript data is available."
                    )
                    return False
                ordered_segments = build_ordered_segments_for_draft(
                    sentences,
                    draft_order,
                    min_duration=self.settings["silence"].get("render_min_duration", 0.6),
                    tail_min_duration=self.settings["silence"].get("tail_min_duration", 0.8),
                    tail_ratio=self.settings["silence"].get("tail_ratio", 0.2),
                    source_range=source_range,
                )
                _save_render_artifacts(
                    "J/L-cut",
                    ordered_segments,
                    overlap_frames,
                    source_range,
                    overlap_jitter,
                    overlap_seed,
                )
                ok = self.bridge.resolve_api.insert_overlapped_segments(
                    clip,
                    ordered_segments,
                    overlap_frames=overlap_frames,
                    overlap_jitter=overlap_jitter,
                    jitter_seed=overlap_seed,
                    include_audio=True,
                )
                return ok
            if draft_order:
                temp_path = self._ensure_unique_path(
                    base_path.with_name(f"{base_path.stem}_reordered{base_path.suffix}")
                )
                logging.getLogger("VideoForge.ui").info("Draft render temp: %s", temp_path)
                render_silence_removed_with_reorder(
                    video_path=video_path,
                    sentences=sentences,
                    draft_order=draft_order,
                    silence_segments=None,
                    output_path=str(temp_path),
                    source_range=source_range,
                    min_duration=self.settings["silence"].get("render_min_duration", 0.6),
                    tail_min_duration=self.settings["silence"].get("tail_min_duration", 0.8),
                    tail_ratio=self.settings["silence"].get("tail_ratio", 0.2),
                    jlcut_mode=jlcut_mode,
                    jcut_offset=jcut_offset,
                    lcut_offset=lcut_offset,
                )
                logging.getLogger("VideoForge.ui").info("Detecting silence on draft render...")
                post_segments = process_silence(
                    video_path=str(temp_path),
                    threshold_db=self.settings["silence"]["threshold_db"],
                    min_keep=self.settings["silence"]["min_keep_duration"],
                    buffer_before=self.settings["silence"]["buffer_before"],
                    buffer_after=self.settings["silence"]["buffer_after"],
                )
                if post_segments:
                    total_keep = sum(float(seg["t1"]) - float(seg["t0"]) for seg in post_segments)
                    logging.getLogger("VideoForge.ui").info(
                        "Draft silence keep: %d segments (total %.2fs)",
                        len(post_segments),
                        total_keep,
                    )
                render_silence_removed(
                    video_path=str(temp_path),
                    segments=post_segments,
                    output_path=str(output_path),
                    source_range=None,
                    min_duration=self.settings["silence"].get("render_min_duration", 0.6),
                )
                try:
                    temp_path.unlink()
                except Exception:
                    logging.getLogger("VideoForge.ui").info(
                        "Draft render temp retained: %s", temp_path
                    )
            else:
                _save_render_artifacts(
                    "Silence Removal",
                    keep_segments,
                    0,
                    source_range,
                )
                render_silence_removed(
                    video_path=video_path,
                    segments=keep_segments,
                    output_path=str(output_path),
                    source_range=source_range,
                    min_duration=self.settings["silence"].get("render_min_duration", 0.6),
                )
            logging.getLogger("VideoForge.ui").info("Silence render output: %s", output_path)
            Config.set("last_silence_render", str(output_path))
            return str(output_path)

        def _done(result):
            if result is True and Config.get("render_mode", "Silence Removal") == "J/L-cut":
                self._set_status("J/L-cut overlap inserted above tracks.")
                return
            if result is False:
                self._set_status("Render failed. See log for details.")
                return
            action = Config.get("silence_action", "Insert Above Track (keep original)")
            if action == "Manual (Media Pool only)":
                try:
                    self.bridge.resolve_api.add_media_to_bin(str(result), "VideoForge Rendered")
                except Exception:
                    pass
            self._set_status(f"Rendered: {Path(result).name}")

        self._run_worker(_render, on_done=_done)

    def _on_replace_with_rendered_clicked(self) -> None:
        clip = self.bridge.resolve_api.get_primary_clip()
        if not clip:
            self._set_status("Select a clip first.")
            return
        rendered = Config.get("last_silence_render")
        if not rendered or not Path(rendered).exists():
            self._set_status("Render file first.")
            return
        action = Config.get("silence_action", "Insert Above Track (keep original)")
        if action == "Manual (Media Pool only)":
            self._set_status("Adding rendered clip to Media Pool...")
        else:
            self._set_status("Inserting rendered clip above selected track...")

        def _replace():
            if action == "Manual (Media Pool only)":
                return self.bridge.resolve_api.add_media_to_bin(rendered, "VideoForge Rendered")
            return self.bridge.resolve_api.insert_rendered_above_clip(
                clip,
                rendered,
                include_audio=True,
            )

        def _done(result):
            if result:
                if action == "Manual (Media Pool only)":
                    self._set_status("Rendered clip added to Media Pool (VideoForge Rendered).")
                else:
                    self._set_status("Rendered clip inserted above (original preserved).")
            else:
                self._set_status("Operation failed. See log.")

        self._run_worker(_replace, on_done=_done)

    def _on_silence_action_changed(self, value: str) -> None:
        Config.set("silence_action", value)

    def _on_jlcut_mode_changed(self, value: str) -> None:
        mode = str(value).strip().lower()
        mode_map = {
            "off": "off",
            "j-cut": "jcut",
            "l-cut": "lcut",
            "both": "both",
            "auto": "auto",
        }
        normalized = mode_map.get(mode, "off")
        self.settings["broll"]["jlcut_mode"] = normalized
        Config.set("jlcut_mode", normalized)

    def _on_jcut_offset_changed(self, value: str) -> None:
        try:
            offset = float(value)
        except Exception:
            return
        self.settings["broll"]["jcut_offset"] = max(0.0, offset)
        Config.set("jcut_offset", float(self.settings["broll"]["jcut_offset"]))

    def _on_lcut_offset_changed(self, value: str) -> None:
        try:
            offset = float(value)
        except Exception:
            return
        self.settings["broll"]["lcut_offset"] = max(0.0, offset)
        Config.set("lcut_offset", float(self.settings["broll"]["lcut_offset"]))

    def _on_jlcut_overlap_changed(self, value: str) -> None:
        try:
            frames = int(float(value))
        except Exception:
            return
        frames = max(0, frames)
        self.settings["broll"]["jlcut_overlap_frames"] = frames
        Config.set("jlcut_overlap_frames", frames)

    def _on_jlcut_overlap_jitter_changed(self, value: str) -> None:
        try:
            frames = int(float(value))
        except Exception:
            return
        frames = max(0, frames)
        self.settings["broll"]["jlcut_overlap_jitter"] = frames
        Config.set("jlcut_overlap_jitter", frames)

    def _on_broll_gen_enabled_changed(self) -> None:
        Config.set("broll_gen_enabled", self.broll_enabled_checkbox.isChecked())

    def _on_broll_output_changed(self, value: str) -> None:
        Config.set("broll_gen_output_dir", value.strip())

    def _on_broll_gen_mode_changed(self, value: str) -> None:
        legacy = str(value).strip().lower()
        Config.set("broll_gen_mode", legacy)
        legacy_map = {
            "stock": "stock_only",
            "ai": "ai_only",
            "hybrid": "library_stock_ai",
        }
        source_mode = legacy_map.get(legacy)
        if source_mode:
            Config.set("broll_source_mode", source_mode)
            self._sync_broll_source_combos(source_mode)

    def _on_broll_output_browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select B-roll Output Folder")
        if folder:
            self.broll_output_edit.setText(folder)
            Config.set("broll_gen_output_dir", folder)

    # --- Event Handlers ---

    def _on_analyze_clicked(self) -> None:
        logging.getLogger("VideoForge.ui").info("Analyze clicked")
        try:
            clip = self.bridge.resolve_api.get_primary_clip()
        except Exception as exc:
            logging.getLogger("VideoForge.ui").error("Failed to read selection: %s", exc)
            self._set_status(f"Failed to read selection: {exc}")
            return
        logging.getLogger("VideoForge.ui").info(
            "Primary clip found: %s", bool(clip)
        )
        if not clip:
            self._set_status("Select a clip or move the playhead over a clip.")
            self.status.setStyleSheet(
                f"color: {COLORS['error']}; font-size: 11px; background: transparent;"
            )
            return
        try:
            video_path = self.bridge._get_clip_path(clip)
        except Exception as exc:
            self._set_status(f"Error: Failed to get clip info: {exc}")
            return
        if not video_path:
            self._set_status("Error: Clip has no file path.")
            return
        clip_range = self.bridge.resolve_api.get_item_range_seconds(clip)
        source_range = self.bridge.resolve_api.get_clip_source_range_seconds(clip)

        engine_raw = self.engine_combo.currentText() if hasattr(self, "engine_combo") else "Whisper"
        engine = engine_raw.strip()
        logging.getLogger("VideoForge.ui").info("Transcription engine selected: %s", engine)
        Config.set("transcription_engine", engine)
        engine_key = engine.lower()
        if engine_key in {"resolve ai", "resolve_ai", "resolve"}:
            language = self.language_combo.currentText() if hasattr(self, "language_combo") else "auto"
            self._set_busy(True)
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self._set_status("Preparing... (1/4)")
            try:
                self._set_progress_value(55)
                self._set_status("Transcribing (Resolve AI)... (2/4)")
                result = self.bridge.analyze_resolve_ai(
                    video_path=video_path,
                    language=language,
                    clip_range=clip_range,
                    source_range=source_range,
                )
                self.project_db = result.get("project_db")
                self.main_video_path = video_path
                self.analyze_status.setText("Status: Analyzed")
                self.analyze_status.setStyleSheet(f"color: {COLORS['success']}; margin-top: 6px;")
                self.edit_transcript_btn.setEnabled(True)
                self._save_project_state()
                self._set_progress_value(90)
                self._set_status("Saving... (3/4)")
                self._set_progress_value(100)
                self._set_status("Complete (4/4)")
            except Exception as exc:
                self._set_status(f"Error: {exc}")
            finally:
                self._set_busy(False)
            return

        def _analyze():
            logger = logging.getLogger("VideoForge.ui.Analyze")
            logger.info("[1/4] Preparing analysis")
            self._update_progress_safe(8)
            self._update_status_safe("Preparing... (1/4)")
            srt_path = self.srt_path_edit.text().strip() if self.srt_path_edit else ""
            srt_path = srt_path or None
            whisper_task = self.settings.get("whisper", {}).get("task", "transcribe")
            whisper_language = self.settings.get("whisper", {}).get("language", "auto")

            logger.info("[2/4] Starting Whisper transcription")
            logger.info("[2/4] video_path: %s", video_path)
            self._update_progress_safe(16)
            self._update_status_safe("Transcribing audio... (2/4) This may take 1-2 minutes")

            result_container = [None]
            error_container = [None]

            def _analyze_with_timeout():
                try:
                    result_container[0] = self.bridge.analyze_from_path(
                        video_path=video_path,
                        whisper_task=whisper_task,
                        whisper_language=whisper_language,
                        align_srt=srt_path,
                        clip_range=source_range,
                    )
                except Exception as exc:
                    error_container[0] = exc

            analyze_thread = threading.Thread(target=_analyze_with_timeout, name="VideoForgeAnalyze")
            analyze_thread.daemon = True
            analyze_thread.start()

            try:
                from VideoForge.config.config_manager import Config

                timeout = int(Config.get("analyze_timeout_sec") or 1200)
            except Exception:
                timeout = 1200
            elapsed = 0
            while analyze_thread.is_alive() and elapsed < timeout:
                analyze_thread.join(timeout=5)
                elapsed += 5
                if elapsed % 15 == 0:
                    logger.info("[2/4] Still transcribing... (%d seconds elapsed)", elapsed)
                    self._update_status_safe(f"Transcribing... ({elapsed}s) (2/4)")
                    self._update_progress_safe(min(90, 16 + int(elapsed / timeout * 70)))

            if analyze_thread.is_alive():
                logger.error("[2/4] TIMEOUT after %d seconds!", timeout)
                raise TimeoutError(
                    f"Analyze timed out after {timeout}s. Try CPU mode or check CUDA/CT2 settings."
                )

            if error_container[0]:
                raise error_container[0]

            result = result_container[0] or {}

            logger.info("[3/4] Saving results to database")
            self._update_progress_safe(95)
            self._update_status_safe("Saving... (3/4)")

            result["main_path"] = video_path

            logger.info("[4/4] Analysis complete")
            self._update_progress_safe(100)
            self._update_status_safe("Complete (4/4)")
            return result

        def _done(result):
            self.project_db = result.get("project_db")
            self.main_video_path = result.get("main_path")
            self.analyze_status.setText("Status: Analyzed")
            self.analyze_status.setStyleSheet(f"color: {COLORS['success']}; margin-top: 6px;")
            self.edit_transcript_btn.setEnabled(True)
            self._save_project_state()
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
        logging.getLogger("VideoForge.ui").info("Browse library clicked")
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

    def _on_browse_global_library(self) -> None:
        logging.getLogger("VideoForge.ui").info("Browse global library clicked")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Global Library Database",
            "",
            "SQLite Database (*.db)",
        )
        if path:
            Config.set("global_library_db_path", path)
            if hasattr(self, "global_library_path_input"):
                self.global_library_path_input.setText(path)

    def _on_browse_local_library(self) -> None:
        logging.getLogger("VideoForge.ui").info("Browse local library clicked")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Local Library Database",
            "",
            "SQLite Database (*.db)",
        )
        if path:
            Config.set("local_library_db_path", path)
            if hasattr(self, "local_library_path_input"):
                self.local_library_path_input.setText(path)

    def _on_library_scope_changed(self, index: int) -> None:
        scope_map = {0: "both", 1: "global", 2: "local"}
        scope = scope_map.get(int(index), "both")
        Config.set("library_search_scope", scope)
        logging.getLogger("VideoForge.ui").info("Library search scope: %s", scope)

    def _on_global_library_path_changed(self, value: str) -> None:
        Config.set("global_library_db_path", str(value or "").strip())

    def _on_local_library_path_changed(self, value: str) -> None:
        Config.set("local_library_db_path", str(value or "").strip())

    def _on_library_path_changed(self, value: str) -> None:
        if hasattr(self, "library_path_display"):
            self.library_path_display.setText(str(value or "").strip())

    def _on_scan_library(self, library_type: str | None = None) -> None:
        if library_type in ("global", "local"):
            self._on_scan_scoped_library(library_type)
            return

        logging.getLogger("VideoForge.ui").info("Scan library clicked")
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

    def _on_scan_scoped_library(self, library_type: str) -> None:
        logging.getLogger("VideoForge.ui").info("Scan %s library clicked", library_type)
        db_key = "global_library_db_path" if library_type == "global" else "local_library_db_path"
        db_path = str(Config.get(db_key, "") or "").strip()
        if not db_path:
            message = f"Set {library_type} library DB path in Settings first."
            self._set_status(message)
            QMessageBox.information(self, "Library Scan", message)
            return
        default_dir = (
            self.saved_broll_dir
            if self.saved_broll_dir and Path(self.saved_broll_dir).exists()
            else ""
        )
        folder = QFileDialog.getExistingDirectory(
            self,
            f"Select {library_type.title()} Library Folder",
            default_dir,
        )
        if not folder:
            return

        def _scan():
            indexer = LibraryIndexer()
            return indexer.index_library(Path(folder), library_type=library_type)

        def _done(result):
            label = "Global" if library_type == "global" else "Local"
            self.library_status.setText(f"Status: Indexed {result} clips ({label})")
            self.library_status.setStyleSheet(f"color: {COLORS['success']}; margin-top: 6px;")
            self._set_status(f"{label} library scan complete")

        self._set_status(f"Scanning {library_type} library...")
        self._run_worker(_scan, on_done=_done)

    def _on_open_library_manager(self) -> None:
        try:
            from VideoForge.ui.library_manager import LibraryManager
        except Exception as exc:
            self._set_status(f"Library Manager unavailable: {exc}")
            return
        global_path = str(Config.get("global_library_db_path") or "").strip()
        local_path = str(Config.get("local_library_db_path") or "").strip()
        db_path = global_path or local_path or self.library_path_edit.text().strip()
        self._library_manager_window = LibraryManager.show_as_dialog(
            self,
            db_path=db_path if db_path else None,
        )
        self._set_status("Opened Library Manager.")

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

    def _on_edit_transcript(self) -> None:
        if not self.project_db:
            self._set_status("Analyze a clip first.")
            return
        try:
            from VideoForge.ui.transcript_editor import TranscriptEditorPanel
        except Exception as exc:
            self._set_status(f"Failed to open editor: {exc}")
            return
        if self.transcript_panel is None:
            self.transcript_panel = TranscriptEditorPanel(self.project_db)
        else:
            try:
                self.transcript_panel.project_db = self.project_db
                self.transcript_panel.store = SegmentStore(self.project_db)
                self.transcript_panel._load_transcript()
            except Exception:
                self.transcript_panel = TranscriptEditorPanel(self.project_db)
        self.transcript_panel.show()

    def _on_match_clicked(self) -> None:
        logging.getLogger("VideoForge.ui").info("Match clicked")
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

    def _run_broll_generation_with_analysis(
        self, sentences: list[dict], library_path: str, use_llm: bool
    ) -> None:
        """Generate B-roll with optional LLM scene analysis (Direct mode)."""
        sentences = [s for s in sentences if str(s.get("text", "")).strip()]
        output_root = self._get_broll_output_root()
        project_name = self._get_broll_project_name()
        max_scenes = self._get_broll_gen_max_scenes()

        # Get API keys from Config or environment
        pexels_key = str(Config.get("pexels_api_key") or "").strip() or os.getenv("PEXELS_API_KEY")
        unsplash_key = str(Config.get("unsplash_api_key") or "").strip() or os.getenv("UNSPLASH_API_KEY")
        comfyui_url = str(Config.get("comfyui_url") or "http://127.0.0.1:8188").strip()

        # ComfyUI workflow path
        repo_root = self._resolve_repo_root()
        workflow_path = repo_root / "VideoForge" / "config" / "z_image_turbo.json"
        if not workflow_path.exists():
            workflow_path = None

        def _generate():
            from VideoForge.broll.direct_generator import DirectBrollGenerator
            from VideoForge.broll.source_config import BrollSourceConfig

            config = BrollSourceConfig.from_config()
            source_mode = str(config.source_mode or "stock_only").strip()
            if source_mode in {"library_stock", "stock_only"}:
                mode = "stock"
            elif source_mode in {"library_ai", "ai_only"}:
                mode = "ai"
            else:
                mode = "hybrid"
            resolution_value = str(Config.get("comfyui_resolution") or "1920x1024")
            seed_value = Config.get("comfyui_seed")
            seed = None
            if seed_value is not None and str(seed_value).strip():
                try:
                    seed = int(str(seed_value).strip())
                except ValueError:
                    seed = None
            try:
                width_str, height_str = resolution_value.lower().split("x", 1)
                comfyui_width = int(width_str.strip())
                comfyui_height = int(height_str.strip())
            except ValueError:
                comfyui_width, comfyui_height = 1920, 1024

            # Step 1: Scene Analysis (LLM or heuristic fallback)
            scene_analyses = None
            selected_sentences = sentences
            selected_ids: set[int] = set()
            try:
                from VideoForge.ai.scene_analyzer import SceneAnalyzer

                analyzer = SceneAnalyzer()
                if max_scenes:
                    scene_analyses = analyzer.select_scenes(sentences, max_scenes)
                    selected_ids = {
                        s.get("sentence_id")
                        for s in (scene_analyses or [])
                        if s.get("sentence_id") is not None
                    }
                    selected_sentences = [
                        s
                        for s in sentences
                        if s.get("id") in selected_ids and str(s.get("text", "")).strip()
                    ]
                    if not selected_sentences:
                        selected_sentences = sentences
                        selected_ids = {
                            s.get("id")
                            for s in selected_sentences
                            if s.get("id") is not None
                        }
                if not scene_analyses:
                    scene_analyses = analyzer.analyze_sentences(selected_sentences, max_scenes)
                if max_scenes:
                    logging.getLogger("VideoForge.ui").info(
                        "Selected %d/%d sentences for B-roll",
                        len(selected_sentences),
                        len(sentences),
                    )
                logging.getLogger("VideoForge.ui").info(
                    "Scene analysis: %d scenes", len(scene_analyses)
                )
            except Exception as exc:
                logging.getLogger("VideoForge.ui").warning(
                    "Scene analysis failed: %s", exc
                )
                # Continue without analysis

            if scene_analyses:
                for scene in scene_analyses:
                    scene["comfyui_width"] = comfyui_width
                    scene["comfyui_height"] = comfyui_height
                    if seed is not None:
                        scene["comfyui_seed"] = seed

            # Step 1.5: LLM prompt generation for ComfyUI (AI modes only)
            ai_modes = {"ai_only", "library_ai", "library_stock_ai"}
            if use_llm and scene_analyses and str(config.source_mode or "") in ai_modes:
                try:
                    from VideoForge.ai.prompt_generator import PromptGenerator

                    ui_logger = logging.getLogger("VideoForge.ui")
                    ui_logger.info(
                        "Generating ComfyUI prompts for %d scenes...",
                        len(scene_analyses),
                    )
                    prompt_gen = PromptGenerator()
                    prompts = prompt_gen.generate_prompts(scene_analyses)
                    prompt_map = {
                        p.get("sentence_id"): p
                        for p in (prompts or [])
                        if isinstance(p, dict) and p.get("sentence_id") is not None
                    }
                    merged = 0
                    for scene in scene_analyses:
                        sid = scene.get("sentence_id")
                        p = prompt_map.get(sid)
                        if not p:
                            continue
                        scene["comfyui_positive"] = p.get("positive")
                        scene["comfyui_negative"] = p.get("negative")
                        if p.get("seed") is not None:
                            scene["comfyui_seed"] = p.get("seed")
                        merged += 1
                    ui_logger.info(
                        "Generated %d ComfyUI prompts (merged=%d)",
                        len(prompts or []),
                        merged,
                    )
                except Exception as exc:
                    logging.getLogger("VideoForge.ui").warning(
                        "Prompt generation failed: %s", exc
                    )

            # Step 2: Direct B-roll Generation
            output_root.mkdir(parents=True, exist_ok=True)
            generator = DirectBrollGenerator(
                output_root,
                pexels_key=pexels_key,
                unsplash_key=unsplash_key,
                comfyui_url=comfyui_url,
                comfyui_workflow=workflow_path,
            )

            selected_sentences = [
                s for s in selected_sentences if str(s.get("text", "")).strip()
            ]
            if not selected_ids:
                selected_ids = {
                    s.get("id") for s in selected_sentences if s.get("id") is not None
                }
            # Step 3: Library-first match (optional)
            schema_path = repo_root / "VideoForge" / "config" / "schema.sql"
            if not schema_path.exists():
                schema_path = repo_root / "config" / "schema.sql"
            library_db = LibraryDB(library_path, schema_path=str(schema_path))
            remaining_sentences = list(selected_sentences)
            if str(config.source_mode or "").startswith("library"):
                from VideoForge.broll.orchestrator import BrollOrchestrator

                orchestrator = BrollOrchestrator(library_db, self.settings)
                library_matches = orchestrator.find_library_matches(selected_sentences)
                if library_matches:
                    logging.getLogger("VideoForge.ui").info(
                        "Library matches found: %d", len(library_matches)
                    )
                remaining_sentences = [
                    s
                    for s in selected_sentences
                    if s.get("id") not in library_matches
                ]

            video_paths = []
            if remaining_sentences and config.source_mode != "library_only":
                video_paths = generator.generate_broll_batch(
                    remaining_sentences,
                    project_name=project_name,
                    mode=mode,
                    scene_analyses=scene_analyses,
                    config=config,
                )
            elif config.source_mode == "library_only" and remaining_sentences:
                logging.getLogger("VideoForge.ui").info(
                    "Source mode is library_only; skipping %d sentences without matches.",
                    len(remaining_sentences),
                )

            from VideoForge.broll.indexer import encode_video_clip

            clip_ids = []
            for video_path in video_paths:
                meta = coerce_metadata_dict(generator.extract_metadata(video_path))
                embedding = encode_video_clip(str(video_path))
                keywords = meta.get("keywords") or []
                visual_elements = meta.get("visual_elements") or []
                if isinstance(keywords, str):
                    keywords = [k for k in keywords.replace(",", " ").split() if k]
                if isinstance(visual_elements, str):
                    visual_elements = [v for v in visual_elements.replace(",", " ").split() if v]
                vision_tokens = list(keywords) + [v for v in visual_elements if v not in keywords]
                description = meta.get("description") or meta.get("generated_from_text") or ""
                created_at = meta.get("created_at", "")
                duration = meta.get("duration")
                if duration is None:
                    duration = 5.0
                db_meta = {
                    "path": str(video_path),
                    "duration": float(duration),
                    "fps": 30.0,  # Default FPS
                    "width": 1920,
                    "height": 1080,
                    "folder_tags": self._build_broll_tags(meta),
                    "vision_tags": ", ".join(vision_tokens),
                    "description": description,
                    "created_at": created_at,
                }
                clip_id = library_db.upsert_clip(db_meta, embedding)
                clip_ids.append(clip_id)

            # Step 4: Re-match
            match_ids = sorted({int(sid) for sid in selected_ids if sid is not None}) if selected_ids else None
            rematch = self.bridge.match_broll(
                self.project_db,
                library_path,
                sentence_ids=match_ids,
            )
            return {"generated": len(clip_ids), "match": rematch}

        def _done(result):
            generated = result.get("generated", 0) if isinstance(result, dict) else 0
            match = result.get("match", {}) if isinstance(result, dict) else {}
            count = match.get("count", 0)
            self._set_status(f"Generated {generated} B-roll clips. Matches: {count}.")

        status_msg = "Generating B-roll"
        if use_llm:
            status_msg += " (with LLM scene analysis)"
        self._set_status(status_msg + "...")
        self._run_worker(_generate, on_done=_done)

    def _build_broll_tags(self, meta: dict) -> str:
        """Build folder tags from generator metadata."""
        raw = meta.get("raw_metadata") if isinstance(meta.get("raw_metadata"), dict) else {}
        source = str(meta.get("source", "") or raw.get("source", "") or "")
        scene_num = str(meta.get("scene_num", "") or raw.get("scene_num", "") or "")
        script_index = str(meta.get("script_index", "") or raw.get("script_index", "") or "")
        keywords = meta.get("keywords", []) or []
        tags = []
        if source:
            tags.append(source)
        if script_index or scene_num:
            tags.append(f"script {script_index or scene_num}")
        if isinstance(keywords, str):
            keywords = [k for k in keywords.replace(",", " ").split() if k]
        tags.extend([str(k) for k in list(keywords)[:5]])
        return " ".join(t for t in tags if t)

    def _on_generate_broll_clicked(self) -> None:
        """Generate B-roll button handler (post-edit workflow)."""
        logging.getLogger("VideoForge.ui").info("Generate B-roll clicked")

        if not self.project_db:
            self._set_status("Analyze a clip first.")
            return

        # Check API keys
        pexels_key = str(Config.get("pexels_api_key") or "").strip() or os.getenv("PEXELS_API_KEY")
        unsplash_key = str(Config.get("unsplash_api_key") or "").strip() or os.getenv("UNSPLASH_API_KEY")

        if not pexels_key and not unsplash_key:
            reply = QMessageBox.question(
                self,
                "API Keys Not Set",
                "Pexels or Unsplash API keys are required for B-roll generation.\n\n"
                "Configure API keys in Settings now?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                # Switch to Settings tab
                self.tab_widget.setCurrentIndex(1)
            return

        # Get sentences (after editor/reorder)
        try:
            store = SegmentStore(self.project_db)
            sentences = store.get_sentences()
            if not sentences:
                self._set_status("No sentences found. Run Analyze first.")
                return
            sentences = [s for s in sentences if str(s.get("text", "")).strip()]
            if not sentences:
                self._set_status("No non-empty sentences found.")
                return
        except Exception as exc:
            self._set_status(f"Failed to load sentences: {exc}")
            return

        # LLM Scene Analysis (optional)
        from VideoForge.ai.llm_hooks import is_llm_enabled

        use_llm_analysis = is_llm_enabled()
        if use_llm_analysis:
            llm_msg = "LLM scene analysis enabled (better B-roll matching)"
        else:
            llm_msg = "LLM scene analysis disabled (basic mode)"

        # Confirm generation
        estimate_count = len(sentences)
        max_scenes = self._get_broll_gen_max_scenes()
        if max_scenes:
            estimate_count = min(estimate_count, max_scenes)
        estimate = estimate_count * 15  # seconds
        source_mode = str(Config.get("broll_source_mode") or "stock_only")
        source_label = self._get_broll_source_label(source_mode)
        reply = QMessageBox.question(
            self,
            "Generate B-roll?",
            f"Generate B-roll for {len(sentences)} sentences?\n\n"
            f"Source: {source_label}\n"
            f"{llm_msg}\n"
            f"Estimated time: ~{estimate} seconds\n\n"
            f"This will create scene clips and auto-index them into your library.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return

        # Get library path
        library_path = self.library_path_edit.text().strip()
        if not library_path:
            self._set_status("Set library path first.")
            return

        # Run B-roll generation with LLM analysis
        self._run_broll_generation_with_analysis(sentences, library_path, use_llm_analysis)

    def _get_broll_gen_max_scenes(self) -> int | None:
        raw = Config.get("broll_max_scenes")
        if raw is None or str(raw).strip() == "":
            raw = Config.get("broll_gen_max_scenes")
        if raw is None or str(raw).strip() == "":
            return None
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return None
        if value <= 0:
            return None
        return value

    def _on_broll_gen_max_changed(self, value: str) -> None:
        value = value.strip()
        if value == "":
            Config.set("broll_gen_max_scenes", None)
            Config.set("broll_max_scenes", None)
            return
        try:
            parsed = int(value)
        except ValueError:
            return
        if parsed <= 0:
            return
        Config.set("broll_gen_max_scenes", parsed)
        Config.set("broll_max_scenes", parsed)

    def _on_broll_source_changed(self, value: str) -> None:
        """B-roll source mode 변경."""
        reverse_map = {
            "Library Only": "library_only",
            "Library + Stock": "library_stock",
            "Library + AI": "library_ai",
            "Library + Stock + AI (Full)": "library_stock_ai",
            "Stock Only": "stock_only",
            "AI Only": "ai_only",
        }
        mode = reverse_map.get(value, "stock_only")
        Config.set("broll_source_mode", mode)
        self._sync_broll_source_combos(mode)

    def _on_broll_media_type_changed(self, value: str) -> None:
        """미디어 타입 선호도 변경."""
        reverse_map = {
            "Auto (Source Default)": "auto",
            "Prefer Video": "prefer_video",
            "Images Only": "image_only",
        }
        mode = reverse_map.get(value, "auto")
        Config.set("broll_prefer_media_type", mode)

    def _sync_broll_source_combos(self, source_mode: str) -> None:
        source_map = {
            "library_only": "Library Only",
            "library_stock": "Library + Stock",
            "library_ai": "Library + AI",
            "library_stock_ai": "Library + Stock + AI (Full)",
            "stock_only": "Stock Only",
            "ai_only": "AI Only",
        }
        label = source_map.get(source_mode, "Stock Only")
        for attr in ("broll_source_combo", "match_source_combo"):
            combo = getattr(self, attr, None)
            if not combo:
                continue
            if combo.currentText() == label:
                continue
            combo.blockSignals(True)
            combo.setCurrentText(label)
            combo.blockSignals(False)

    @staticmethod
    def _get_broll_source_label(source_mode: str) -> str:
        source_map = {
            "library_only": "Library Only",
            "library_stock": "Library + Stock",
            "library_ai": "Library + AI",
            "library_stock_ai": "Library + Stock + AI (Full)",
            "stock_only": "Stock Only",
            "ai_only": "AI Only",
        }
        return source_map.get(source_mode, "Stock Only")

    def _on_reset_project_broll_clicked(self) -> None:
        if not self.project_db:
            self._set_status("Analyze a clip first.")
            return
        library_path = self.library_path_edit.text().strip()
        if not library_path:
            self._set_status("Set library path first.")
            return

        project_name = self._get_broll_project_name()
        output_root = self._get_broll_output_root()
        project_dir = output_root / project_name

        reply = QMessageBox.question(
            self,
            "Reset Project B-roll?",
            "This will remove generated B-roll clips for the current project,\n"
            "clear project matches, and remove those clips from the library.\n\n"
            "Library DB itself will not be deleted.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        def _reset():
            removed_files = 0
            if project_dir.exists():
                removed_files = len([p for p in project_dir.rglob("*") if p.is_file()])
                shutil.rmtree(project_dir, ignore_errors=True)

            from VideoForge.core.segment_store import SegmentStore

            store = SegmentStore(self.project_db)
            store.clear_matches()

            removed_db = 0
            schema_path = self._resolve_repo_root() / "VideoForge" / "config" / "schema.sql"
            if not schema_path.exists():
                schema_path = self._resolve_repo_root() / "config" / "schema.sql"
            library_db = LibraryDB(library_path, schema_path=str(schema_path))
            prefixes = {str(project_dir), str(project_dir.resolve())}
            for prefix in prefixes:
                removed_db += library_db.delete_clips_by_path_prefix(prefix)

            return {"files": removed_files, "db": removed_db}

        def _done(result):
            removed_files = result.get("files", 0) if isinstance(result, dict) else 0
            removed_db = result.get("db", 0) if isinstance(result, dict) else 0
            self._set_status(
                f"Reset complete: removed {removed_files} files, {removed_db} library clips."
            )

        self._set_status("Resetting project B-roll...")
        self._run_worker(_reset, on_done=_done)


    def _on_broll_min_scenes_changed(self, value: str) -> None:
        """Minimum scenes for LLM selection."""
        value = value.strip()
        if not value:
            Config.set("broll_min_scenes", None)
            return
        try:
            parsed = int(value)
        except ValueError:
            return
        if parsed <= 0:
            Config.set("broll_min_scenes", None)
            return
        Config.set("broll_min_scenes", parsed)

    def _on_broll_max_scenes_changed(self, value: str) -> None:
        """최대 장면 개수 변경."""
        try:
            num = int(value)
            if 1 <= num <= 50:
                Config.set("broll_max_scenes", num)
                Config.set("broll_gen_max_scenes", num)
        except ValueError:
            return

    def _on_broll_scene_seed_changed(self, value: str) -> None:
        """Scene selection seed value."""
        value = value.strip()
        if not value:
            Config.set("broll_scene_seed", None)
            return
        try:
            parsed = int(value)
        except ValueError:
            return
        Config.set("broll_scene_seed", parsed)

    def _on_apply_clicked(self) -> None:
        logging.getLogger("VideoForge.ui").info("Apply clicked")
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

    def _on_cleanup_projects(self) -> None:
        projects_dir = (
            self._resolve_repo_root() / "VideoForge" / "data" / "projects"
        )
        keep_latest = int(Config.get("cleanup_keep_latest", 5))
        min_age_days = int(Config.get("cleanup_min_age_days", 14))

        def _cleanup():
            return cleanup_project_files(
                projects_dir,
                self.project_db,
                keep_latest=keep_latest,
                min_age_days=min_age_days,
            )

        def _done(result):
            self._set_status(
                f"Cleanup complete: removed {result.removed}, kept {result.kept}, "
                f"skipped {result.skipped_recent}."
            )

        self._set_status("Cleaning old projects...")
        self._run_worker(_cleanup, on_done=_done)

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

    def _start_resolve_watchdog(self) -> None:
        self._resolve_watchdog = QTimer(self)
        self._resolve_watchdog.setInterval(3000)
        self._resolve_watchdog.timeout.connect(self._check_resolve_alive)
        self._resolve_watchdog.start()

    def _check_resolve_alive(self) -> None:
        try:
            if not self.bridge.resolve_api.is_available():
                logging.getLogger("VideoForge.ui").info("Resolve API unavailable; closing panel.")
                self.close()
                QApplication.quit()
                return
        except Exception:
            return
        try:
            self._update_draft_render_emphasis()
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
