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

    panel.browse_btn = QPushButton("Browse")
    panel.browse_btn.setFixedWidth(36)
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

    panel.library_status = QLabel("Status: Library not scanned")
    panel.library_status.setObjectName("StatusLabel")
    card_layout.addWidget(panel.library_status)

    parent_layout.addWidget(card)
