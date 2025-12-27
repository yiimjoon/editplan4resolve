"""Library section builder."""

from __future__ import annotations

from VideoForge.config.config_manager import Config
from VideoForge.ui.qt_compat import QHBoxLayout, QLabel, QLineEdit, QPushButton, Qt, QVBoxLayout


def build_library_section(panel, parent_layout: QVBoxLayout) -> None:
    """Build the B-roll library card."""
    card = panel._create_card()
    card_layout = QVBoxLayout(card)
    card_layout.setSpacing(8)

    card_layout.addWidget(panel._create_section_title("2. B-roll Library"))

    card_layout.addSpacing(6)

    scope_label = QLabel("Global / Local Library")
    scope_label.setStyleSheet(f"color: {panel.colors['text_dim']}; font-size: 11px;")
    card_layout.addWidget(scope_label)

    global_row = QHBoxLayout()
    global_label = QLabel("Global Library (Shared)")
    global_label.setStyleSheet(f"color: {panel.colors['text_dim']}; font-size: 11px;")
    panel.scan_global_library_btn = QPushButton("Scan Global Library")
    panel.scan_global_library_btn.setCursor(Qt.PointingHandCursor)
    panel.scan_global_library_btn.clicked.connect(
        lambda _checked=False: panel._on_scan_library("global")
    )
    global_row.addWidget(global_label)
    global_row.addStretch()
    global_row.addWidget(panel.scan_global_library_btn)
    card_layout.addLayout(global_row)

    global_scan_layout = QHBoxLayout()
    panel.global_scan_path_input = QLineEdit()
    panel.global_scan_path_input.setText(str(Config.get("global_library_scan_dir", "")))
    panel.global_scan_path_input.setPlaceholderText("Global scan folder...")
    panel.global_scan_path_input.textChanged.connect(panel._on_global_scan_path_changed)
    global_scan_layout.addWidget(panel.global_scan_path_input)
    panel.global_scan_browse_btn = QPushButton("Browse...")
    panel.global_scan_browse_btn.clicked.connect(panel._on_browse_global_scan_folder)
    global_scan_layout.addWidget(panel.global_scan_browse_btn)
    card_layout.addLayout(global_scan_layout)

    local_row = QHBoxLayout()
    local_label = QLabel("Local Library (Project)")
    local_label.setStyleSheet(f"color: {panel.colors['text_dim']}; font-size: 11px;")
    panel.scan_local_library_btn = QPushButton("Scan Local Library")
    panel.scan_local_library_btn.setCursor(Qt.PointingHandCursor)
    panel.scan_local_library_btn.clicked.connect(
        lambda _checked=False: panel._on_scan_library("local")
    )
    local_row.addWidget(local_label)
    local_row.addStretch()
    local_row.addWidget(panel.scan_local_library_btn)
    card_layout.addLayout(local_row)

    local_scan_layout = QHBoxLayout()
    panel.local_scan_path_input = QLineEdit()
    panel.local_scan_path_input.setText(str(Config.get("local_library_scan_dir", "")))
    panel.local_scan_path_input.setPlaceholderText("Local scan folder...")
    panel.local_scan_path_input.textChanged.connect(panel._on_local_scan_path_changed)
    local_scan_layout.addWidget(panel.local_scan_path_input)
    panel.local_scan_browse_btn = QPushButton("Browse...")
    panel.local_scan_browse_btn.clicked.connect(panel._on_browse_local_scan_folder)
    local_scan_layout.addWidget(panel.local_scan_browse_btn)
    card_layout.addLayout(local_scan_layout)

    panel.library_status = QLabel("Status: Library not scanned")
    panel.library_status.setObjectName("StatusLabel")
    card_layout.addWidget(panel.library_status)

    parent_layout.addWidget(card)
