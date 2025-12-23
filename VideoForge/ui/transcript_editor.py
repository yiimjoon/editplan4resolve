"""Transcript editor panel for VideoForge results."""

from __future__ import annotations

from copy import deepcopy
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
        QFrame,
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
        QFrame,
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

STYLESHEET = f"""
    QWidget {{
        background-color: {COLORS['bg_window']};
        font-family: 'Segoe UI', 'Open Sans', sans-serif;
        font-size: 13px;
        color: {COLORS['text_main']};
    }}
    QTableWidget {{
        background-color: #1a1a1a;
        gridline-color: {COLORS['button_bg']};
        border: 1px solid {COLORS['button_bg']};
        border-radius: 4px;
        selection-background-color: #4a4a4a;
    }}
    QHeaderView::section {{
        background-color: {COLORS['button_bg']};
        padding: 4px;
        border: 1px solid {COLORS['border']};
        color: {COLORS['text_dim']};
        font-weight: bold;
    }}
    QTableWidget::item {{
        padding: 4px;
    }}
    QTableWidget::item:hover {{
        background-color: #3a3a3a;
    }}
    QPushButton {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3a3a3a, stop:1 #2e2e2e);
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 6px 12px;
        color: {COLORS['text_main']};
        border-top: 1px solid {COLORS['highlight']};
        border-left: 1px solid {COLORS['highlight']};
    }}
    QPushButton:hover {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #454545, stop:1 #353535);
    }}
    QPushButton:pressed {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #252525, stop:1 #303030);
    }}
    QPushButton#PrimaryButton {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {COLORS['accent_hover']}, stop:1 {COLORS['accent']});
        color: #1a1a1a;
        font-weight: bold;
        border-top: 1px solid #ffae57;
        border-left: 1px solid #ffae57;
    }}
    QPushButton#PrimaryButton:hover {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffaf60, stop:1 #e69138);
    }}
    QLineEdit {{
        background-color: #1a1a1a;
        border: 1px solid {COLORS['button_bg']};
        border-radius: 6px;
        padding: 4px;
        color: {COLORS['text_main']};
        border-top: 1px solid #0a0a0a;
        border-left: 1px solid #0a0a0a;
    }}
"""


class TranscriptEditorPanel(QWidget):
    """Editable transcript viewer backed by the project DB."""

    def __init__(self, project_db_path: str) -> None:
        super().__init__()
        self.setStyleSheet(STYLESHEET)
        self.project_db = project_db_path
        self.store = SegmentStore(project_db_path)
        self.sentences: List[dict] = []
        self._history: List[List[dict]] = []
        self.resolve_api = ResolveAPI()
        self.logger = logging.getLogger("VideoForge.ui.Transcript")
        self._build_ui()
        self._load_transcript()

    def _build_ui(self) -> None:
        self.setWindowTitle("VideoForge - Transcript Editor")
        self.setMinimumSize(800, 500)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header = QHBoxLayout()
        self.source_label = QLabel("Source: Loading...")
        self.source_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-weight: bold;")
        header.addWidget(self.source_label)
        header.addStretch()

        # Group 1: Manage
        manage_layout = QHBoxLayout()
        reload_btn = QPushButton("Reload")
        reload_btn.setCursor(Qt.PointingHandCursor)
        reload_btn.clicked.connect(self._load_transcript)
        manage_layout.addWidget(reload_btn)

        save_btn = QPushButton("Save Changes")
        save_btn.setObjectName("PrimaryButton")
        save_btn.setCursor(Qt.PointingHandCursor)
        save_btn.clicked.connect(self._save_transcript)
        manage_layout.addWidget(save_btn)
        header.addLayout(manage_layout)
        
        header.addSpacing(20)

        # Group 2: Export
        export_layout = QHBoxLayout()
        export_btn = QPushButton("Export to Resolve")
        export_btn.setCursor(Qt.PointingHandCursor)
        export_btn.clicked.connect(self._export_to_resolve)
        export_layout.addWidget(export_btn)

        export_txt_btn = QPushButton("TXT")
        export_txt_btn.setCursor(Qt.PointingHandCursor)
        export_txt_btn.clicked.connect(self._export_txt)
        export_layout.addWidget(export_txt_btn)

        export_json_btn = QPushButton("JSON")
        export_json_btn.setCursor(Qt.PointingHandCursor)
        export_json_btn.clicked.connect(self._export_json)
        export_layout.addWidget(export_json_btn)
        header.addLayout(export_layout)

        layout.addLayout(header)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"background-color: {COLORS['button_bg']};")
        line.setFixedHeight(1)
        layout.addWidget(line)

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

        options_row.addSpacing(20)
        self.split_btn = QPushButton("✂️ Split Long Lines")
        self.split_btn.setCursor(Qt.PointingHandCursor)
        self.split_btn.setToolTip("Automatically split segments that exceed max chars")
        self.split_btn.clicked.connect(self._on_split_clicked)
        options_row.addWidget(self.split_btn)

        self.undo_btn = QPushButton("Undo")
        self.undo_btn.setCursor(Qt.PointingHandCursor)
        self.undo_btn.setToolTip("Undo last transcript edit")
        self.undo_btn.clicked.connect(self._on_undo_clicked)
        options_row.addWidget(self.undo_btn)

        self.merge_btn = QPushButton("Merge Selected")
        self.merge_btn.setCursor(Qt.PointingHandCursor)
        self.merge_btn.setToolTip("Merge selected consecutive rows")
        self.merge_btn.clicked.connect(self._on_merge_clicked)
        options_row.addWidget(self.merge_btn)

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

    def _load_transcript(self, reset_history: bool = True) -> None:
        self.logger.info("Transcript load: %s", self.project_db)
        self.sentences = self.store.get_sentences()
        if reset_history:
            self._history = [deepcopy(self.sentences)]
        model = self.store.get_artifact("whisper_model") or "unknown"
        self.source_label.setText(f"Source: {model}")
        try:
            self.max_chars_edit.setText(str(Config.get("subtitle_max_chars", 42)))
        except Exception:
            pass

        self._render_table()
        self.logger.info("Transcript loaded: %d sentences", len(self.sentences))

    def _save_transcript(self) -> None:
        self._push_history("save")
        for i in range(self.table.rowCount()):
            text_item = self.table.item(i, 3)
            if text_item:
                self.sentences[i]["text"] = text_item.text()
        self.store.update_sentences(self.sentences)
        self._load_transcript()
        self.stats_label.setText("Changes saved to database.")

    def _on_split_clicked(self) -> None:
        max_chars = self._parse_int(self.max_chars_edit.text(), 42)
        gap_frames = self._parse_int(self.gap_frames_edit.text(), 0)
        try:
            fps = self.resolve_api.get_timeline_fps()
        except Exception:
            fps = 24.0
        gap_sec = float(gap_frames) / float(fps) if gap_frames > 0 else 0.0
        changed = False
        self._push_history("split")

        replaced = 0
        for s in list(self.sentences):
            text = s.get("text", "").strip()
            if len(text) <= max_chars:
                continue

            changed = True
            # Use existing wrap logic but discard max_lines for splitting
            chunks = self._wrap_text(text, max_chars, 0)
            if not chunks:
                continue

            t0, t1 = float(s.get("t0", 0.0)), float(s.get("t1", 0.0))
            duration = t1 - t0
            weights = [max(1, len(c)) for c in chunks]
            total = float(sum(weights))
            if duration <= 0.0 or total <= 0.0:
                continue

            gap_total = gap_sec * (len(chunks) - 1)
            if gap_total >= duration:
                gap_sec = 0.0
                gap_total = 0.0
            available = max(0.0, duration - gap_total)
            curr_t0 = t0
            created: List[dict] = []
            for idx, chunk in enumerate(chunks):
                if idx == len(chunks) - 1:
                    curr_t1 = t1
                else:
                    ratio = float(weights[idx]) / total
                    curr_t1 = curr_t0 + (available * ratio)
                new_seg = {
                    "segment_id": s.get("segment_id"),
                    "t0": float(curr_t0),
                    "t1": float(curr_t1),
                    "text": chunk,
                    "confidence": s.get("confidence"),
                    "metadata": {**(s.get("metadata") or {}), "split_from": s.get("id")},
                }
                created.append(new_seg)
                curr_t0 = curr_t1 + gap_sec

            if s.get("id") is None:
                continue
            self.store.replace_sentence(int(s["id"]), created)
            replaced += 1

        if changed:
            self._load_transcript(reset_history=False)
            self._history.append(deepcopy(self.sentences))
            self.stats_label.setText(f"Split {replaced} long segments (> {max_chars} chars).")
        else:
            self._history.pop()

    def _on_undo_clicked(self) -> None:
        if len(self._history) < 2:
            self.stats_label.setText("Nothing to undo.")
            return
        self._history.pop()
        snapshot = deepcopy(self._history[-1])
        self.store.replace_all_sentences(snapshot)
        self.sentences = snapshot
        self._render_table()
        self.stats_label.setText("Undo applied.")

    def _on_merge_clicked(self) -> None:
        selected_rows = sorted({idx.row() for idx in self.table.selectionModel().selectedRows()})
        if len(selected_rows) < 2:
            self.stats_label.setText("Select 2+ consecutive rows to merge.")
            return
        expected = list(range(selected_rows[0], selected_rows[0] + len(selected_rows)))
        if selected_rows != expected:
            self.stats_label.setText("Select consecutive rows only.")
            return

        self._push_history("merge")
        first = self.sentences[selected_rows[0]]
        last = self.sentences[selected_rows[-1]]
        merged_text = " ".join(
            str(self.sentences[i].get("text", "")).strip()
            for i in selected_rows
        ).strip()
        merged = {
            "segment_id": first.get("segment_id"),
            "t0": float(first.get("t0", 0.0)),
            "t1": float(last.get("t1", 0.0)),
            "text": merged_text,
            "confidence": first.get("confidence"),
            "metadata": {
                **(first.get("metadata") or {}),
                "merged_from": [self.sentences[i].get("id") for i in selected_rows],
            },
        }

        new_sentences: List[dict] = []
        for idx, sentence in enumerate(self.sentences):
            if idx == selected_rows[0]:
                new_sentences.append(merged)
                continue
            if idx in selected_rows:
                continue
            new_sentences.append(sentence)

        self.store.replace_all_sentences(new_sentences)
        self.sentences = new_sentences
        self._render_table()
        self._history.append(deepcopy(self.sentences))
        self.stats_label.setText(f"Merged {len(selected_rows)} rows.")

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

    def _render_table(self) -> None:
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

    def _push_history(self, reason: str) -> None:
        if not self.sentences:
            return
        self.logger.info("History snapshot saved (%s).", reason)
        self._history.append(deepcopy(self.sentences))

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
            return TranscriptEditorPanel._fix_korean_line_breaks(lines, max_chars)
        return TranscriptEditorPanel._fix_korean_line_breaks(lines, max_chars)

    @staticmethod
    def _fix_korean_line_breaks(lines: List[str], max_chars: int) -> List[str]:
        """
        Heuristics to avoid awkward Korean splits like:
          "거기 걸려버린" / "거예요"
        This does not require any LLM and only rearranges word boundaries.
        """

        def normalize_token(token: str) -> str:
            return token.strip().strip(".,!?…\"'”“’‘()[]{}")

        suffix_words = {
            "거예요",
            "거에요",
            "거죠",
            "거야",
            "거지",
            "거",
            "요",
            "죠",
            "네",
            "예",
            "음",
        }

        # Common Korean particles/endings that should not start a new line alone.
        suffix_prefixes = (
            "은",
            "는",
            "이",
            "가",
            "을",
            "를",
            "에",
            "에서",
            "으로",
            "로",
            "와",
            "과",
            "도",
            "만",
            "까지",
            "부터",
            "처럼",
            "보다",
            "조차",
            "마저",
            "라도",
            "이나",
            "든지",
            "랑",
            "하고",
            "께",
            "에게",
            "한테",
            "입니다",
            "습니다",
            "했다",
            "했다면",
            "한다",
            "한다면",
            "된다",
            "돼요",
        )

        fixed = list(lines)
        changed = True
        while changed:
            changed = False
            for i in range(1, len(fixed)):
                prev = fixed[i - 1].strip()
                curr = fixed[i].strip()
                if not prev or not curr:
                    continue

                curr_words = curr.split()
                prev_words = prev.split()
                if not curr_words or not prev_words:
                    continue

                first_norm = normalize_token(curr_words[0])
                curr_is_suffixy = False
                if len(curr_words) == 1:
                    curr_is_suffixy = first_norm in suffix_words or first_norm.startswith(suffix_prefixes)
                else:
                    curr_is_suffixy = first_norm in suffix_words

                if not curr_is_suffixy:
                    continue

                # If we can merge the entire current line into previous, do it.
                merged = f"{prev} {curr}".strip()
                if len(merged) <= max_chars:
                    fixed[i - 1] = merged
                    fixed.pop(i)
                    changed = True
                    break

                # Otherwise, move one word from previous to current to avoid suffix-only lines.
                if len(prev_words) >= 2:
                    candidate_word = prev_words[-1]
                    new_prev = " ".join(prev_words[:-1]).strip()
                    new_curr = f"{candidate_word} {curr}".strip()
                    if new_prev and len(new_prev) <= max_chars and len(new_curr) <= max_chars:
                        fixed[i - 1] = new_prev
                        fixed[i] = new_curr
                        changed = True
                        break

        # Drop any empty lines that might have been created.
        return [line for line in fixed if line.strip()]

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
