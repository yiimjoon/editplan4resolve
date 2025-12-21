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


class Worker(QObject):
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
        layout = QVBoxLayout()

        layout.addWidget(QLabel("1. Select and Analyze Main Clip"))
        self.analyze_btn = QPushButton("Analyze Selected Clip")
        self.analyze_btn.clicked.connect(self._on_analyze_clicked)
        layout.addWidget(self.analyze_btn)
        self.analyze_status = QLabel("Status: Not analyzed")
        layout.addWidget(self.analyze_status)

        layout.addWidget(QLabel("\n2. B-roll Library"))
        library_layout = QHBoxLayout()
        self.library_path_edit = QLineEdit()
        self.library_path_edit.setPlaceholderText("Library DB path...")
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self._on_browse_library)
        library_layout.addWidget(self.library_path_edit)
        library_layout.addWidget(self.browse_btn)
        layout.addLayout(library_layout)

        self.scan_library_btn = QPushButton("Scan Library Folder")
        self.scan_library_btn.clicked.connect(self._on_scan_library)
        layout.addWidget(self.scan_library_btn)
        self.library_status = QLabel("Status: Library not scanned")
        layout.addWidget(self.library_status)

        layout.addWidget(QLabel("\n3. Match B-roll and Apply"))
        self.threshold_label = QLabel("Matching Threshold: 0.65")
        layout.addWidget(self.threshold_label)
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(65)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        layout.addWidget(self.threshold_slider)

        self.match_btn = QPushButton("Match B-roll")
        self.match_btn.clicked.connect(self._on_match_clicked)
        layout.addWidget(self.match_btn)

        self.apply_btn = QPushButton("Apply to Timeline")
        self.apply_btn.clicked.connect(self._on_apply_clicked)
        layout.addWidget(self.apply_btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status = QLabel("Ready")
        layout.addWidget(self.status)

        self.setLayout(layout)
        self.setWindowTitle("VideoForge - Auto B-roll")

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
            self._set_status(f"Operation failed: {err}")
        elif on_done:
            on_done(result)
        thread.quit()
        thread.wait()
        if thread in self._threads:
            self._threads.remove(thread)

    def _on_analyze_clicked(self) -> None:
        def _analyze():
            project_db = self.bridge.analyze_selected_clip()
            clips = self.bridge.resolve_api.get_selected_clips()
            main_path = clips[0].GetClipProperty("File Path") if clips else None
            return project_db, main_path

        def _done(result):
            self.project_db, self.main_video_path = result
            self.analyze_status.setText("Status: Analyzed")
            self._set_status("Analyze complete")

        self._set_status("Analyzing... (silence + speech)")
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
                library_path = str(self._resolve_repo_root() / "VideoForge" / "data" / "cache" / "library.db")
                self.library_path_edit.setText(library_path)
            db = LibraryDB(
                library_path,
                schema_path=str(self._resolve_repo_root() / "VideoForge" / "config" / "schema.sql"),
            )
            count = scan_library(folder, db)
            return count

        def _done(result):
            self.library_status.setText(f"Status: Indexed {result} clips")
            self._set_status("Library scan complete")
            self.bridge.save_library_path(self.library_path_edit.text().strip())

        self._set_status("Scanning library...")
        self._run_worker(_scan, on_done=_done)

    def _on_threshold_changed(self, value: int) -> None:
        threshold = value / 100.0
        self.threshold_label.setText(f"Matching Threshold: {threshold:.2f}")

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
            self._set_status("Applied to timeline")

        self._set_status("Applying to timeline... (placing clips)")
        self._run_worker(_apply, on_done=_done)

    def _load_saved_library_path(self) -> None:
        try:
            saved = self.bridge.get_saved_library_path()
        except Exception:
            saved = None
        if saved:
            self.library_path_edit.setText(saved)
