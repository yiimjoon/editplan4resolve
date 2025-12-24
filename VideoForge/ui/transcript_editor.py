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
from VideoForge.core.transcript_utils import split_sentence, wrap_text
from VideoForge.ai.llm_hooks import (
    get_llm_instructions,
    get_llm_instructions_mode,
    is_llm_enabled,
    suggest_reflow_lines,
    suggest_reorder,
    write_llm_env,
)
from VideoForge.ui.qt_compat import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHeaderView,
    QInputDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    Qt,
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

    MAX_HISTORY = 50

    def __init__(self, project_db_path: str) -> None:
        super().__init__()
        self.setStyleSheet(STYLESHEET)
        self.project_db = project_db_path
        self.store = SegmentStore(project_db_path)
        self.sentences: List[dict] = []
        self._history: List[List[dict]] = []
        self._redo: List[List[dict]] = []
        self.resolve_api = ResolveAPI()
        self.logger = logging.getLogger("VideoForge.ui.Transcript")
        self._draft_active = False
        self._draft_original_count = 0
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
        self.export_resolve_btn = QPushButton("Export to Resolve")
        self.export_resolve_btn.setCursor(Qt.PointingHandCursor)
        self.export_resolve_btn.clicked.connect(self._export_to_resolve)
        export_layout.addWidget(self.export_resolve_btn)

        self.export_txt_btn = QPushButton("TXT")
        self.export_txt_btn.setCursor(Qt.PointingHandCursor)
        self.export_txt_btn.clicked.connect(self._export_txt)
        export_layout.addWidget(self.export_txt_btn)

        self.export_json_btn = QPushButton("JSON")
        self.export_json_btn.setCursor(Qt.PointingHandCursor)
        self.export_json_btn.clicked.connect(self._export_json)
        export_layout.addWidget(self.export_json_btn)
        header.addLayout(export_layout)

        header.addSpacing(12)
        self.instructions_mode_combo = QComboBox()
        self.instructions_mode_combo.addItems(["shortform", "longform"])
        current_mode = str(Config.get("llm_instructions_mode") or get_llm_instructions_mode())
        if current_mode in {"shortform", "longform"}:
            self.instructions_mode_combo.setCurrentText(current_mode)
        else:
            self.instructions_mode_combo.setCurrentText("shortform")
        self.instructions_mode_combo.currentTextChanged.connect(self._on_instructions_mode_changed)
        header.addWidget(self.instructions_mode_combo)

        instructions_btn = QPushButton("LLM Instructions")
        instructions_btn.setCursor(Qt.PointingHandCursor)
        instructions_btn.setToolTip("Edit optional LLM guidance")
        instructions_btn.clicked.connect(self._on_edit_instructions)
        header.addWidget(instructions_btn)

        layout.addLayout(header)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"background-color: {COLORS['button_bg']};")
        line.setFixedHeight(1)
        layout.addWidget(line)

        self.draft_banner = QLabel("Draft Order Active (Reordered view)")
        self.draft_banner.setStyleSheet(
            "background-color: #3a2a1a; color: #f5c26b; padding: 6px; border-radius: 4px;"
        )
        self.draft_banner.setVisible(False)
        layout.addWidget(self.draft_banner)

        draft_controls = QHBoxLayout()
        self.clear_draft_btn = QPushButton("Clear Draft")
        self.clear_draft_btn.setCursor(Qt.PointingHandCursor)
        self.clear_draft_btn.clicked.connect(self._on_clear_draft)
        draft_controls.addWidget(self.clear_draft_btn)

        self.apply_draft_btn = QPushButton("Apply Draft to Original")
        self.apply_draft_btn.setCursor(Qt.PointingHandCursor)
        self.apply_draft_btn.clicked.connect(self._on_apply_draft)
        draft_controls.addWidget(self.apply_draft_btn)
        draft_controls.addStretch()
        layout.addLayout(draft_controls)

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
        self.split_btn = QPushButton("Split Long Lines")
        self.split_btn.setCursor(Qt.PointingHandCursor)
        self.split_btn.setToolTip("Automatically split segments that exceed max chars")
        self.split_btn.clicked.connect(self._on_split_clicked)
        options_row.addWidget(self.split_btn)

        self.undo_btn = QPushButton("Undo")
        self.undo_btn.setCursor(Qt.PointingHandCursor)
        self.undo_btn.setToolTip("Undo last transcript edit")
        self.undo_btn.clicked.connect(self._on_undo_clicked)
        options_row.addWidget(self.undo_btn)

        self.redo_btn = QPushButton("Redo")
        self.redo_btn.setCursor(Qt.PointingHandCursor)
        self.redo_btn.setToolTip("Redo last transcript edit")
        self.redo_btn.clicked.connect(self._on_redo_clicked)
        options_row.addWidget(self.redo_btn)

        self.merge_btn = QPushButton("Merge Selected")
        self.merge_btn.setCursor(Qt.PointingHandCursor)
        self.merge_btn.setToolTip("Merge selected consecutive rows")
        self.merge_btn.clicked.connect(self._on_merge_clicked)
        options_row.addWidget(self.merge_btn)

        self.ai_reflow_btn = QPushButton("AI Reflow (Draft)")
        self.ai_reflow_btn.setCursor(Qt.PointingHandCursor)
        self.ai_reflow_btn.setToolTip("Use external LLM to reflow lines (requires provider)")
        self.ai_reflow_btn.setEnabled(is_llm_enabled())
        self.ai_reflow_btn.clicked.connect(self._on_ai_reflow_clicked)
        options_row.addWidget(self.ai_reflow_btn)

        self.ai_reorder_btn = QPushButton("AI Reorder (Draft)")
        self.ai_reorder_btn.setCursor(Qt.PointingHandCursor)
        self.ai_reorder_btn.setToolTip("Use LLM to reorder transcript (draft workflow)")
        self.ai_reorder_btn.setEnabled(is_llm_enabled())
        self.ai_reorder_btn.clicked.connect(self._on_ai_reorder_clicked)
        options_row.addWidget(self.ai_reorder_btn)

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
        prev_draft_active = bool(getattr(self, "_draft_active", False))
        self.sentences = self.store.get_sentences()
        updated = self.store.ensure_sentence_uids()
        if updated:
            self.logger.info("Transcript uid update: %d rows", updated)
            self.sentences = self.store.get_sentences()
        self._draft_original_count = len(self.sentences)
        self._draft_active = self.store.has_draft_order()
        if prev_draft_active != self._draft_active:
            self.logger.info("Draft mode changed: %s", self._draft_active)
        if self._draft_active:
            self.sentences = self._apply_draft_order(self.sentences)
        if reset_history:
            self._history = [deepcopy(self.sentences)]
            self._redo = []
        model = self.store.get_artifact("whisper_model") or "unknown"
        self.source_label.setText(f"Source: {model}")
        try:
            self.max_chars_edit.setText(str(Config.get("subtitle_max_chars", 42)))
        except Exception:
            pass
        try:
            self.ai_reflow_btn.setEnabled(is_llm_enabled())
        except Exception:
            pass

        self._render_table()
        self._update_draft_ui()
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
            created = split_sentence(s, max_chars, gap_sec)
            if not created or created == [s]:
                continue

            if s.get("id") is None:
                continue
            self.store.replace_sentence(int(s["id"]), created)
            replaced += 1

        if changed:
            self.store.clear_draft_order()
            self._load_transcript(reset_history=False)
            self._history.append(deepcopy(self.sentences))
            self._redo = []
            self.stats_label.setText(f"Split {replaced} long segments (> {max_chars} chars).")
        else:
            self._history.pop()

    def _on_undo_clicked(self) -> None:
        if len(self._history) < 2:
            self.stats_label.setText("Nothing to undo.")
            return
        self._redo.append(deepcopy(self._history.pop()))
        snapshot = deepcopy(self._history[-1])
        self.store.replace_all_sentences(snapshot)
        self.sentences = snapshot
        self._render_table()
        self.stats_label.setText("Undo applied.")

    def _on_redo_clicked(self) -> None:
        if not self._redo:
            self.stats_label.setText("Nothing to redo.")
            return
        snapshot = deepcopy(self._redo.pop())
        self._history.append(deepcopy(snapshot))
        self.store.replace_all_sentences(snapshot)
        self.sentences = snapshot
        self._render_table()
        self.stats_label.setText("Redo applied.")

    def _on_clear_draft(self) -> None:
        if not self.store.has_draft_order():
            self.stats_label.setText("Draft mode is not active.")
            return
        try:
            order_len = len(self.store.get_draft_order())
        except Exception:
            order_len = -1
        self.logger.info("Draft clear requested (order_len=%s).", order_len)
        self.store.clear_draft_order()
        self._load_transcript(reset_history=False)
        self.stats_label.setText("Draft order cleared.")
        self.logger.info("Draft cleared.")

    def _on_apply_draft(self) -> None:
        if not self.store.has_draft_order():
            self.stats_label.setText("Draft mode is not active.")
            return
        choice = QMessageBox.question(
            self,
            "Apply Draft to Original",
            "This will overwrite the original transcript order and recalculate timestamps.\n\n"
            "Timing will no longer match the source audio until you re-render/re-transcribe.\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if choice != QMessageBox.Yes:
            self.stats_label.setText("Draft apply canceled.")
            self.logger.info("Draft apply canceled by user.")
            return
        try:
            fps = self.resolve_api.get_timeline_fps()
        except Exception:
            fps = 24.0
        gap_frames = self._parse_int(self.gap_frames_edit.text(), 0)
        gap_sec = float(gap_frames) / float(fps) if gap_frames > 0 else 0.0
        self.logger.info("Draft apply started (gap_frames=%d, fps=%.3f).", gap_frames, fps)

        base_sentences = self.store.get_sentences()
        ordered = self._apply_draft_order(base_sentences)
        cursor = 0.0
        applied: List[dict] = []
        for sentence in ordered:
            duration = float(sentence.get("t1", 0.0)) - float(sentence.get("t0", 0.0))
            if duration <= 0:
                continue
            updated = deepcopy(sentence)
            updated["t0"] = cursor
            updated["t1"] = cursor + duration
            applied.append(updated)
            cursor = updated["t1"] + gap_sec
        if not applied:
            self.stats_label.setText("Draft apply produced no valid rows.")
            self.logger.warning("Draft apply produced no valid rows.")
            return
        self.store.replace_all_sentences(applied)
        self.store.clear_draft_order()
        self._load_transcript(reset_history=False)
        self.stats_label.setText("Draft applied to original transcript.")
        self.logger.info("Draft applied (rows=%d).", len(applied))

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
        self.store.clear_draft_order()
        self.sentences = new_sentences
        self._render_table()
        self._history.append(deepcopy(self.sentences))
        self._redo = []
        self.stats_label.setText(f"Merged {len(selected_rows)} rows.")

    def _on_ai_reflow_clicked(self) -> None:
        if not is_llm_enabled():
            self.stats_label.setText("LLM provider not configured.")
            return
        max_chars = self._parse_int(self.max_chars_edit.text(), 42)
        try:
            reflowed = suggest_reflow_lines(self.sentences, max_chars)
        except NotImplementedError as exc:
            self.stats_label.setText(str(exc))
            self.logger.warning("AI reflow not configured: %s", exc)
            return
        except Exception as exc:
            self.stats_label.setText(f"AI reflow failed: {exc}")
            self.logger.error("AI reflow failed: %s", exc, exc_info=True)
            return

        if not reflowed:
            self.stats_label.setText("AI reflow produced no changes.")
            return
        self._push_history("ai_reflow")
        self.store.replace_all_sentences(reflowed)
        self.store.clear_draft_order()
        self.sentences = reflowed
        self._render_table()
        self._history.append(deepcopy(self.sentences))
        self._redo = []
        self.stats_label.setText("AI reflow applied.")

    def _on_ai_reorder_clicked(self) -> None:
        if not is_llm_enabled():
            self.stats_label.setText("LLM provider not configured.")
            return

        profile = Config.get("llm_reorder_profile", "hook_build_payoff")
        cut_strength = float(Config.get("llm_reorder_cut_strength", 0.2))
        try:
            result = suggest_reorder(self.sentences, profile=profile, cut_strength=cut_strength)
        except NotImplementedError as exc:
            self.stats_label.setText(str(exc))
            self.logger.warning("AI reorder not configured: %s", exc)
            return
        except Exception as exc:
            self.stats_label.setText(f"AI reorder failed: {exc}")
            self.logger.error("AI reorder failed: %s", exc, exc_info=True)
            return

        order = result.get("order") or []
        drops = set(result.get("drops") or [])
        if not order:
            self.stats_label.setText("AI reorder produced no changes.")
            return

        invalid_indices = 0
        ordered_uids: List[str] = []
        for idx in order:
            if idx in drops:
                continue
            if idx < 1 or idx > len(self.sentences):
                invalid_indices += 1
                continue
            metadata = self.sentences[idx - 1].get("metadata") or {}
            uid = metadata.get("uid")
            if not uid:
                invalid_indices += 1
                continue
            ordered_uids.append(uid)

        if not ordered_uids:
            self.stats_label.setText("AI reorder produced an empty transcript.")
            return

        self.store.save_draft_order(ordered_uids)
        self.sentences = self._apply_draft_order(self.sentences)
        self._render_table()
        self.stats_label.setText("AI reorder saved as draft order.")
        if invalid_indices:
            self.logger.warning(
                "AI reorder ignored %d invalid indices (valid range: 1-%d).",
                invalid_indices,
                len(self.sentences),
            )

    def _on_edit_instructions(self) -> None:
        mode = str(Config.get("llm_instructions_mode") or get_llm_instructions_mode())
        current = str(get_llm_instructions() or Config.get("llm_instructions", ""))
        text, ok = QInputDialog.getMultiLineText(
            self,
            "LLM Instructions",
            "Optional guidance for LLM actions:",
            current,
        )
        if not ok:
            return
        if mode == "longform":
            Config.set("llm_instructions_longform", text)
        else:
            Config.set("llm_instructions_shortform", text)
        Config.set("llm_instructions_mode", mode)
        write_llm_env(instructions=text, instructions_mode=mode)
        self.stats_label.setText("LLM instructions updated.")

    def _on_instructions_mode_changed(self, value: str) -> None:
        mode = str(value).strip().lower()
        if mode not in {"shortform", "longform"}:
            return
        Config.set("llm_instructions_mode", mode)
        write_llm_env(instructions_mode=mode)
        self.stats_label.setText(f"LLM mode set: {mode}")

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
            chunks = wrap_text(text, max_chars, 0)
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
            chunks = wrap_text(text, max_chars, max_lines)
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
        if self._draft_active:
            self.stats_label.setText(
                f"Draft Mode Active | Total: {len(self.sentences)} (Original: {self._draft_original_count})"
                f" | Duration: {self._format_time(duration)}"
            )
        else:
            self.stats_label.setText(
                f"Total: {len(self.sentences)} sentences | Duration: {self._format_time(duration)}"
            )

    def _update_draft_ui(self) -> None:
        active = self._draft_active
        try:
            self.draft_banner.setVisible(active)
        except Exception:
            pass
        try:
            self.clear_draft_btn.setEnabled(active)
            self.apply_draft_btn.setEnabled(active)
        except Exception:
            pass
        self._set_table_style(active)
        self._set_button_primary(getattr(self, "export_resolve_btn", None), active)
        self._set_button_primary(getattr(self, "export_txt_btn", None), active)
        self._set_button_primary(getattr(self, "export_json_btn", None), active)

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

    def _set_table_style(self, draft_active: bool) -> None:
        if draft_active:
            self.table.setStyleSheet(
                "QTableWidget {"
                "background-color: #3a2a1a;"
                "gridline-color: #4a3a2a;"
                "border: 1px solid #4a3a2a;"
                "border-radius: 4px;"
                "selection-background-color: #55402f;"
                "}"
                "QTableWidget::item:hover { background-color: #4a3727; }"
            )
        else:
            self.table.setStyleSheet(
                "QTableWidget {"
                "background-color: #1a1a1a;"
                "gridline-color: #3e3e3e;"
                "border: 1px solid #3e3e3e;"
                "border-radius: 4px;"
                "selection-background-color: #4a4a4a;"
                "}"
                "QTableWidget::item:hover { background-color: #3a3a3a; }"
            )

    def _push_history(self, reason: str) -> None:
        if not self.sentences:
            return
        self.logger.info("History snapshot saved (%s).", reason)
        self._history.append(deepcopy(self.sentences))
        if len(self._history) > self.MAX_HISTORY:
            self._history.pop(0)

    def _apply_draft_order(self, sentences: List[dict]) -> List[dict]:
        order = self.store.get_draft_order()
        if not order:
            return sentences
        by_uid = {}
        for sentence in sentences:
            metadata = sentence.get("metadata") or {}
            uid = metadata.get("uid")
            if uid:
                by_uid[uid] = sentence
        ordered = []
        for uid in order:
            if uid in by_uid:
                ordered.append(by_uid.pop(uid))
        ordered.extend(by_uid.values())
        return ordered

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
