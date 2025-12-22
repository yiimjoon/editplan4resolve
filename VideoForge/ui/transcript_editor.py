"""Transcript editor panel for VideoForge results."""

from __future__ import annotations

from pathlib import Path
from typing import List

from VideoForge.core.segment_store import SegmentStore

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QHeaderView,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except Exception:
    from PySide2.QtCore import Qt
    from PySide2.QtWidgets import (
        QHeaderView,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )


class TranscriptEditorPanel(QWidget):
    """Editable transcript viewer backed by the project DB."""

    def __init__(self, project_db_path: str) -> None:
        super().__init__()
        self.project_db = project_db_path
        self.store = SegmentStore(project_db_path)
        self.sentences: List[dict] = []
        self._build_ui()
        self._load_transcript()

    def _build_ui(self) -> None:
        self.setWindowTitle("VideoForge - Transcript Editor")
        self.setMinimumSize(720, 420)

        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        self.source_label = QLabel("Source: Loading...")
        header.addWidget(self.source_label)
        header.addStretch()

        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self._load_transcript)
        header.addWidget(reload_btn)

        save_btn = QPushButton("Save Changes")
        save_btn.clicked.connect(self._save_transcript)
        header.addWidget(save_btn)

        layout.addLayout(header)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["#", "Start", "End", "Text"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.DoubleClicked)
        layout.addWidget(self.table)

        self.stats_label = QLabel("Total: 0 sentences")
        layout.addWidget(self.stats_label)

    def _load_transcript(self) -> None:
        self.sentences = self.store.get_sentences()
        model = self.store.get_artifact("whisper_model") or "unknown"
        self.source_label.setText(f"Source: {model}")

        self.table.setRowCount(len(self.sentences))
        for i, sentence in enumerate(self.sentences):
            self.table.setItem(i, 0, self._readonly_item(str(i + 1)))
            self.table.setItem(i, 1, self._readonly_item(self._format_time(sentence.get("t0", 0.0))))
            self.table.setItem(i, 2, self._readonly_item(self._format_time(sentence.get("t1", 0.0))))
            text_item = QTableWidgetItem(sentence.get("text", ""))
            text_item.setFlags(text_item.flags() | Qt.ItemIsEditable)
            self.table.setItem(i, 3, text_item)

        duration = self.sentences[-1]["t1"] if self.sentences else 0.0
        self.stats_label.setText(
            f"Total: {len(self.sentences)} sentences | Duration: {self._format_time(duration)}"
        )

    def _save_transcript(self) -> None:
        for i in range(self.table.rowCount()):
            text_item = self.table.item(i, 3)
            if text_item:
                self.sentences[i]["text"] = text_item.text()
        self.store.update_sentences(self.sentences)
        self._load_transcript()

    @staticmethod
    def _readonly_item(text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        return item

    @staticmethod
    def _format_time(seconds: float) -> str:
        mins = int(seconds // 60)
        secs = float(seconds % 60)
        return f"{mins}:{secs:05.2f}"
