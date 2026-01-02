"""Analyze section builder."""

from __future__ import annotations

from VideoForge.ui.qt_compat import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    Qt,
    QVBoxLayout,
    QTextBrowser,
)
from VideoForge.ui.widgets.collapsible_card import CollapsibleCard


def build_analyze_section(panel, parent_layout: QVBoxLayout) -> None:
    """Build the Transcribe page cards."""
    selection_card = panel._create_card()
    selection_layout = QVBoxLayout(selection_card)
    selection_layout.setSpacing(8)
    selection_layout.addWidget(panel._create_section_title("1. Clip Selection"))

    desc = QLabel("Select a clip in the timeline. The current selection is shown below.")
    desc.setObjectName("Description")
    desc.setWordWrap(True)
    selection_layout.addWidget(desc)

    selection_row = QHBoxLayout()
    panel.clip_selection_label = QLabel("Selected: -")
    panel.clip_selection_label.setStyleSheet(
        f"color: {panel.colors['text_dim']}; font-size: 11px; background: transparent;"
    )
    selection_row.addWidget(panel.clip_selection_label)
    selection_row.addStretch()
    panel.refresh_clip_btn = QPushButton("Refresh Selection")
    panel.refresh_clip_btn.setCursor(Qt.PointingHandCursor)
    panel.refresh_clip_btn.clicked.connect(panel._on_refresh_clip_selection)
    selection_row.addWidget(panel.refresh_clip_btn)
    selection_layout.addLayout(selection_row)
    parent_layout.addWidget(selection_card)

    settings_card = CollapsibleCard(
        "2. Transcription Settings", section_id="transcribe_settings", collapsed=True
    )
    settings_layout = settings_card.content_layout
    panel.transcribe_settings_summary = QLabel("Engine: Whisper | Language: auto | Task: transcribe")
    panel.transcribe_settings_summary.setStyleSheet(
        f"color: {panel.colors['text_dim']}; font-size: 11px; background: transparent;"
    )
    settings_layout.addWidget(panel.transcribe_settings_summary)
    panel.open_settings_btn = QPushButton("Open Settings")
    panel.open_settings_btn.setCursor(Qt.PointingHandCursor)
    panel.open_settings_btn.clicked.connect(panel._open_settings_page)
    settings_layout.addWidget(panel.open_settings_btn)
    parent_layout.addWidget(settings_card)

    action_card = panel._create_card()
    action_layout = QVBoxLayout(action_card)
    action_layout.setSpacing(8)
    action_layout.addWidget(panel._create_section_title("3. Action"))

    panel.analyze_btn = QPushButton("Analyze Selected Clip")
    panel.analyze_btn.setObjectName("PrimaryButton")
    panel.analyze_btn.setCursor(Qt.PointingHandCursor)
    panel.analyze_btn.setFixedHeight(36)
    panel.analyze_btn.clicked.connect(panel._on_analyze_clicked)
    action_layout.addWidget(panel.analyze_btn)

    panel.analyze_status = QLabel("Status: Not analyzed")
    panel.analyze_status.setObjectName("StatusLabel")
    action_layout.addWidget(panel.analyze_status)

    log_row = QHBoxLayout()
    log_row.setSpacing(6)
    panel.log_path_label = QLabel("Log: VideoForge.log")
    log_path = panel._get_log_path()
    if log_path:
        panel.log_path_label.setToolTip(str(log_path))
    panel.log_path_label.setStyleSheet(
        f"color: {panel.colors['text_dim']}; font-size: 11px; background: transparent;"
    )
    panel.open_log_btn = QPushButton("Open Log File")
    panel.open_log_btn.setCursor(Qt.PointingHandCursor)
    panel.open_log_btn.clicked.connect(panel._on_open_log)
    log_row.addWidget(panel.log_path_label)
    log_row.addStretch()
    log_row.addWidget(panel.open_log_btn)
    action_layout.addLayout(log_row)
    parent_layout.addWidget(action_card)

    results_card = panel._create_card()
    results_layout = QVBoxLayout(results_card)
    results_layout.setSpacing(8)
    results_layout.addWidget(panel._create_section_title("4. Results"))

    panel.transcript_preview = QTextBrowser()
    panel.transcript_preview.setReadOnly(True)
    panel.transcript_preview.setMinimumHeight(140)
    panel.transcript_preview.setText("No transcript loaded.")
    results_layout.addWidget(panel.transcript_preview)

    panel.edit_transcript_btn = QPushButton("Edit Transcript")
    panel.edit_transcript_btn.setCursor(Qt.PointingHandCursor)
    panel.edit_transcript_btn.setEnabled(False)
    panel.edit_transcript_btn.clicked.connect(panel._on_edit_transcript)
    results_layout.addWidget(panel.edit_transcript_btn)

    parent_layout.addWidget(results_card)
