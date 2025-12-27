"""Audio sync section builder."""

from __future__ import annotations

from VideoForge.ui.qt_compat import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    Qt,
    QVBoxLayout,
)


def build_sync_section(panel, parent_layout: QVBoxLayout) -> None:
    """Build the Audio-based Sync card."""
    card = panel._create_card()
    card_layout = QVBoxLayout(card)
    card_layout.setSpacing(8)

    card_layout.addWidget(panel._create_section_title("3. Audio-based Sync"))

    ref_label = QLabel("Reference Video (A-roll):")
    ref_label.setStyleSheet(f"color: {panel.colors['text_dim']}; font-size: 11px;")
    card_layout.addWidget(ref_label)

    ref_layout = QHBoxLayout()
    panel.sync_ref_label = QLabel("No file selected")
    panel.sync_ref_label.setStyleSheet(
        f"color: {panel.colors['text_dim']}; font-size: 10px;"
    )
    panel.sync_ref_browse_btn = QPushButton("Browse...")
    panel.sync_ref_browse_btn.clicked.connect(panel._on_browse_sync_reference)
    ref_layout.addWidget(panel.sync_ref_label, 1)
    ref_layout.addWidget(panel.sync_ref_browse_btn)
    card_layout.addLayout(ref_layout)

    tgt_label = QLabel("Target Videos (B-roll / Multi-camera):")
    tgt_label.setStyleSheet(f"color: {panel.colors['text_dim']}; font-size: 11px;")
    card_layout.addWidget(tgt_label)

    panel.sync_target_list = QListWidget()
    panel.sync_target_list.setMaximumHeight(100)
    panel.sync_target_list.setStyleSheet("font-size: 10px;")
    card_layout.addWidget(panel.sync_target_list)

    tgt_btn_layout = QHBoxLayout()
    panel.sync_add_btn = QPushButton("Add Files...")
    panel.sync_add_btn.clicked.connect(panel._on_add_sync_targets)
    panel.sync_clear_btn = QPushButton("Clear")
    panel.sync_clear_btn.clicked.connect(panel.sync_target_list.clear)
    tgt_btn_layout.addWidget(panel.sync_add_btn)
    tgt_btn_layout.addWidget(panel.sync_clear_btn)
    card_layout.addLayout(tgt_btn_layout)

    mode_label = QLabel("Sync Mode:")
    mode_label.setStyleSheet(f"color: {panel.colors['text_dim']}; font-size: 11px;")
    card_layout.addWidget(mode_label)

    panel.sync_mode_combo = QComboBox()
    panel.sync_mode_combo.addItems(
        [
            "Same Pattern (Multi-camera)",
            "Inverse Pattern (A-roll <-> B-roll)",
        ]
    )
    card_layout.addWidget(panel.sync_mode_combo)

    panel.auto_sync_btn = QPushButton("Auto Sync")
    panel.auto_sync_btn.setCursor(Qt.PointingHandCursor)
    panel.auto_sync_btn.clicked.connect(panel._on_auto_sync_clicked)
    card_layout.addWidget(panel.auto_sync_btn)

    panel.sync_status = QLabel("Status: Ready")
    panel.sync_status.setStyleSheet(
        f"color: {panel.colors['text_dim']}; font-size: 11px;"
    )
    card_layout.addWidget(panel.sync_status)

    parent_layout.addWidget(card)
