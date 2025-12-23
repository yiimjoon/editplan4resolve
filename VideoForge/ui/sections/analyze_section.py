"""Analyze section builder."""

from __future__ import annotations

from VideoForge.ui.qt_compat import QHBoxLayout, QLabel, QPushButton, Qt, QVBoxLayout


def build_analyze_section(panel, parent_layout: QVBoxLayout) -> None:
    """Build the Analyze card."""
    card = panel._create_card()
    card_layout = QVBoxLayout(card)
    card_layout.setSpacing(8)

    card_layout.addWidget(panel._create_section_title("1. Analyze Main Clip"))

    desc = QLabel("Select a clip in the timeline, then analyze its content.")
    desc.setObjectName("Description")
    desc.setWordWrap(True)
    card_layout.addWidget(desc)

    panel.analyze_btn = QPushButton("Analyze Selected Clip")
    panel.analyze_btn.setObjectName("PrimaryButton")
    panel.analyze_btn.setCursor(Qt.PointingHandCursor)
    panel.analyze_btn.setFixedHeight(36)
    panel.analyze_btn.clicked.connect(panel._on_analyze_clicked)
    card_layout.addWidget(panel.analyze_btn)

    panel.analyze_status = QLabel("Status: Not analyzed")
    panel.analyze_status.setObjectName("StatusLabel")
    card_layout.addWidget(panel.analyze_status)

    panel.edit_transcript_btn = QPushButton("Edit Transcript")
    panel.edit_transcript_btn.setCursor(Qt.PointingHandCursor)
    panel.edit_transcript_btn.setEnabled(False)
    panel.edit_transcript_btn.clicked.connect(panel._on_edit_transcript)
    card_layout.addWidget(panel.edit_transcript_btn)

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
    card_layout.addLayout(log_row)

    parent_layout.addWidget(card)
