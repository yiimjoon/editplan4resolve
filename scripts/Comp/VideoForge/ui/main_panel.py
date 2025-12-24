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
    QObject,
    QProgressBar,
    QPushButton,
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
from VideoForge.ui.sections.settings_section import build_settings_section


# --- Color Palette (DaVinci Resolve Style) ---
COLORS = {
    "bg_window": "#252525",
    "bg_card": "#2e2e2e",
    "text_main": "#ececec",
    "text_dim": "#9e9e9e",
    "accent": "#da8a35",
    "accent_hover": "#e69b4e",
    "button_bg": "#3e3e3e",
    "button_hover": "#4e4e4e",
    "border": "#1a1a1a",
    "highlight": "#3a3a3a",
    "success": "#4caf50",
    "error": "#f44336",
}

VERSION = "v1.5.8"

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
        logging.getLogger("VideoForge.ui").info("UI init done (%s)", VERSION)

    def _init_ui(self) -> None:
        """Initialize the modernized UI."""
        self.setStyleSheet(STYLESHEET)
        self.setWindowTitle("VideoForge - Auto B-roll")
        self.resize(860, 520)
        self.setMinimumSize(720, 420)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        # --- Header (Always Visible) ---
        self._setup_header(layout)

        # --- Tabs ---
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Tab 1: Analyze
        analyze_tab = QWidget()
        analyze_layout = QVBoxLayout(analyze_tab)
        analyze_layout.setContentsMargins(4, 8, 4, 4)
        build_analyze_section(self, analyze_layout)
        build_library_section(self, analyze_layout)
        analyze_layout.addStretch()
        self.tabs.addTab(analyze_tab, "📝 Analyze")

        # Tab 2: Match
        match_tab = QWidget()
        match_layout = QVBoxLayout(match_tab)
        match_layout.setContentsMargins(4, 8, 4, 4)
        build_match_section(self, match_layout)
        match_layout.addStretch()
        self.tabs.addTab(match_tab, "🎬 Match")

        # Tab 3: Settings
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        settings_layout.setContentsMargins(4, 8, 4, 4)
        build_settings_section(self, settings_layout)
        settings_layout.addStretch()
        self.tabs.addTab(settings_tab, "⚙️ Settings")

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
        for key in ("jcut_offset", "lcut_offset", "jlcut_overlap_frames"):
            override = Config.get(key)
            if override is None:
                continue
            try:
                if key == "jlcut_overlap_frames":
                    broll[key] = int(float(override))
                else:
                    broll[key] = float(override)
            except (TypeError, ValueError):
                continue

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
            Config.set("llm_provider", "")
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
        self.transcript_chars_label.setVisible(show)
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
            keep_segments = process_silence(
                video_path=video_path,
                threshold_db=self.settings["silence"]["threshold_db"],
                min_keep=self.settings["silence"]["min_keep_duration"],
                buffer_before=self.settings["silence"]["buffer_before"],
                buffer_after=self.settings["silence"]["buffer_after"],
            )
            sentences = []
            draft_order = None
            if self.project_db:
                store = SegmentStore(self.project_db)
                sentences = store.get_sentences()
                if store.has_draft_order():
                    draft_order = store.get_draft_order()
                    logging.getLogger("VideoForge.ui").info(
                        "Silence render using draft order (%d entries).",
                        len(draft_order),
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
                ok = self.bridge.resolve_api.insert_overlapped_segments(
                    clip,
                    ordered_segments,
                    overlap_frames=overlap_frames,
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

            timeout = 600
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

    def _on_scan_library(self) -> None:
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
