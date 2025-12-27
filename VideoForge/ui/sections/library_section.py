"""Library section builder."""

from __future__ import annotations

from VideoForge.ui.qt_compat import QHBoxLayout, QLabel, QLineEdit, QPushButton, Qt, QVBoxLayout


def build_library_section(panel, parent_layout: QVBoxLayout) -> None:
    """Build the B-roll library card."""
    card = panel._create_card()
    card_layout = QVBoxLayout(card)
    card_layout.setSpacing(8)

    card_layout.addWidget(panel._create_section_title("2. B-roll Library"))

    path_layout = QHBoxLayout()
    path_layout.setSpacing(6)

    panel.library_path_edit = QLineEdit()
    panel.library_path_edit.setPlaceholderText("Library DB path...")
    panel.library_path_edit.textChanged.connect(panel._on_library_path_changed)

    panel.open_log_btn = QPushButton("Open Log")
    panel.browse_btn = QPushButton("...")
    panel.browse_btn.setFixedWidth(30)
    panel.browse_btn.setToolTip("Browse for library DB file")
    panel.browse_btn.setCursor(Qt.PointingHandCursor)
    panel.browse_btn.clicked.connect(panel._on_browse_library)

    path_layout.addWidget(panel.library_path_edit)
    path_layout.addWidget(panel.browse_btn)
    card_layout.addLayout(path_layout)

    panel.scan_library_btn = QPushButton("Scan Library Folder")
    panel.scan_library_btn.setCursor(Qt.PointingHandCursor)
    panel.scan_library_btn.clicked.connect(panel._on_scan_library)
    card_layout.addWidget(panel.scan_library_btn)

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

    panel.library_status = QLabel("Status: Library not scanned")
    panel.library_status.setObjectName("StatusLabel")
    card_layout.addWidget(panel.library_status)

    parent_layout.addWidget(card)
