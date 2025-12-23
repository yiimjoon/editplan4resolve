"""Transcript editor panel for VideoForge results."""

from __future__ import annotations

from pathlib import Path
from typing import List

import json
import logging

from VideoForge.core.segment_store import SegmentStore
from VideoForge.config.config_manager import Config
from VideoForge.integrations.resolve_api import ResolveAPI

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QFileDialog,
        QHeaderView,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except Exception:
    from PySide2.QtCore import Qt
    from PySide2.QtWidgets import (
        QFileDialog,
        QHeaderView,
        QHBoxLayout,
        QLabel,
        QLineEdit,
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
        self.resolve_api = ResolveAPI()
        self.logger = logging.getLogger("VideoForge.ui.Transcript")
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

        export_btn = QPushButton("Export to Resolve")
        export_btn.clicked.connect(self._export_to_resolve)
        header.addWidget(export_btn)

        export_txt_btn = QPushButton("Export TXT")
        export_txt_btn.clicked.connect(self._export_txt)
        header.addWidget(export_txt_btn)

        export_json_btn = QPushButton("Export JSON")
        export_json_btn.clicked.connect(self._export_json)
        header.addWidget(export_json_btn)

        layout.addLayout(header)

        options_row = QHBoxLayout()
        options_row.addWidget(QLabel("Max chars per line"))
        self.max_chars_edit = QLineEdit()
        self.max_chars_edit.setFixedWidth(60)
        self.max_chars_edit.setText(str(Config.get("subtitle_max_chars", 42)))
        options_row.addWidget(self.max_chars_edit)

        options_row.addSpacing(12)
        options_row.addWidget(QLabel("Gap (frames)"))
        self.gap_frames_edit = QLineEdit()
        self.gap_frames_edit.setFixedWidth(60)
        self.gap_frames_edit.setText(str(Config.get("subtitle_gap_frames", 0)))
        options_row.addWidget(self.gap_frames_edit)

        options_row.addStretch()
        layout.addLayout(options_row)

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
        self.logger.info("Transcript load: %s", self.project_db)
        self.sentences = self.store.get_sentences()
        model = self.store.get_artifact("whisper_model") or "unknown"
        self.source_label.setText(f"Source: {model}")
        try:
            self.max_chars_edit.setText(str(Config.get("subtitle_max_chars", 42)))
        except Exception:
            pass

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
        self.logger.info("Transcript loaded: %d sentences", len(self.sentences))

    def _save_transcript(self) -> None:
        for i in range(self.table.rowCount()):
            text_item = self.table.item(i, 3)
            if text_item:
                self.sentences[i]["text"] = text_item.text()
        self.store.update_sentences(self.sentences)
        self._load_transcript()

    def _export_to_resolve(self) -> None:
        if not self.sentences:
            self.logger.info("Export skipped: no sentences to export.")
            return
        max_chars = self._parse_int(self.max_chars_edit.text(), 42)
        max_lines = 1
        gap_frames = self._parse_int(self.gap_frames_edit.text(), 0)
        Config.set("subtitle_max_chars", max_chars)
        Config.set("subtitle_max_lines", max_lines)
        Config.set("subtitle_gap_frames", gap_frames)

        srt_path = self._write_srt(max_chars, max_lines, gap_frames)
        if not srt_path:
            return
        ok = self.resolve_api.import_subtitles_into_timeline(srt_path)
        if ok:
            self.logger.info("Exported subtitles to Resolve: %s", srt_path)
            self.stats_label.setText("Exported subtitles to Resolve.")
        else:
            self.logger.warning("Resolve subtitle import failed: %s", srt_path)
            self.stats_label.setText("Export failed. Check log for details.")

    def _export_txt(self) -> None:
        if not self.sentences:
            self.logger.info("TXT export skipped: no sentences to export.")
            return
        max_chars = self._parse_int(self.max_chars_edit.text(), 42)
        default_path = str(Path(self.project_db).with_suffix(".txt"))
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Transcript TXT",
            default_path,
            "Text Files (*.txt)",
        )
        if not path:
            return
        lines = []
        for sentence in self.sentences:
            t0 = self._format_srt_time(float(sentence.get("t0", 0.0)))
            t1 = self._format_srt_time(float(sentence.get("t1", 0.0)))
            text = str(sentence.get("text", "")).strip()
            chunks = self._wrap_text(text, max_chars, 0)
            if not chunks:
                continue
            if len(chunks) == 1:
                lines.append(f"[{t0} - {t1}] {chunks[0]}")
            else:
                lines.append(f"[{t0} - {t1}]")
                lines.extend(chunks)
        Path(path).write_text("\n".join(lines), encoding="utf-8")
        self.stats_label.setText("Exported TXT.")
        self.logger.info("Exported TXT: %s", path)

    def _export_json(self) -> None:
        if not self.sentences:
            self.logger.info("JSON export skipped: no sentences to export.")
            return
        default_path = str(Path(self.project_db).with_suffix(".json"))
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Transcript JSON",
            default_path,
            "JSON Files (*.json)",
        )
        if not path:
            return
        data = {
            "project_db": self.project_db,
            "count": len(self.sentences),
            "sentences": self.sentences,
        }
        Path(path).write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.stats_label.setText("Exported JSON.")
        self.logger.info("Exported JSON: %s", path)

    def _write_srt(self, max_chars: int, max_lines: int, gap_frames: int) -> str | None:
        try:
            fps = self.resolve_api.get_timeline_fps()
        except Exception:
            fps = 24.0
        gap_sec = float(gap_frames) / float(fps) if gap_frames > 0 else 0.0

        out_path = Path(self.project_db).with_suffix(".srt")
        lines: List[str] = []
        subtitle_index = 1
        for sentence in self.sentences:
            t0 = float(sentence.get("t0", 0.0))
            t1 = float(sentence.get("t1", 0.0))
            text = str(sentence.get("text", ""))
            chunks = self._wrap_text(text, max_chars, max_lines)
            if not chunks:
                continue
            if len(chunks) == 1:
                if gap_sec > 0:
                    t1 = max(t0, t1 - gap_sec)
                lines.append(str(subtitle_index))
                lines.append(f"{self._format_srt_time(t0)} --> {self._format_srt_time(t1)}")
                lines.extend(chunks)
                lines.append("")
                subtitle_index += 1
                continue

            total = max(0.0, t1 - t0)
            gap_total = gap_sec * (len(chunks) - 1)
            if gap_total >= total:
                gap_sec = 0.0
                gap_total = 0.0
            slice_duration = (total - gap_total) / float(len(chunks)) if len(chunks) else total
            current_start = t0
            for chunk in chunks:
                current_end = current_start + slice_duration
                lines.append(str(subtitle_index))
                lines.append(
                    f"{self._format_srt_time(current_start)} --> {self._format_srt_time(current_end)}"
                )
                lines.append(chunk)
                lines.append("")
                subtitle_index += 1
                current_start = current_end + gap_sec
        out_path.write_text("\r\n".join(lines), encoding="utf-8-sig")
        self.logger.info("Wrote SRT: %s (%d entries)", out_path, subtitle_index - 1)
        return str(out_path)

    @staticmethod
    def _wrap_text(text: str, max_chars: int, max_lines: int) -> List[str]:
        words = text.replace("\n", " ").split()
        if not words:
            return []
        lines: List[str] = []
        current = ""
        for word in words:
            if not current:
                current = word
                continue
            if len(current) + 1 + len(word) <= max_chars:
                current = f"{current} {word}"
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        if max_lines > 0:
            return lines
        return lines

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        total_ms = int(round(seconds * 1000.0))
        ms = total_ms % 1000
        total_sec = total_ms // 1000
        s = total_sec % 60
        total_min = total_sec // 60
        m = total_min % 60
        h = total_min // 60
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    @staticmethod
    def _parse_int(value: str, default: int) -> int:
        try:
            return int(str(value).strip())
        except Exception:
            return default

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
