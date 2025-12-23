"""Match section builder."""

from __future__ import annotations

from VideoForge.ui.qt_compat import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    Qt,
    QVBoxLayout,
)


def build_match_section(panel, parent_layout: QVBoxLayout) -> None:
    """Build the Match & Apply card."""
    card = panel._create_card()
    card_layout = QVBoxLayout(card)
    card_layout.setSpacing(8)

    card_layout.addWidget(panel._create_section_title("3. Match & Apply"))

    slider_header = QHBoxLayout()
    slider_label = QLabel("Matching Threshold")
    slider_label.setStyleSheet("background: transparent;")

    panel.threshold_label = QLabel(
        f"{int(panel.settings['broll']['matching_threshold'] * 100)}%"
    )
    panel.threshold_label.setStyleSheet(
        f"color: {panel.colors['accent']}; font-weight: bold; background: transparent;"
    )
    panel.threshold_label.setAlignment(Qt.AlignRight)

    slider_header.addWidget(slider_label)
    slider_header.addWidget(panel.threshold_label)
    card_layout.addLayout(slider_header)

    panel.threshold_slider = QSlider(Qt.Horizontal)
    panel.threshold_slider.setMinimum(0)
    panel.threshold_slider.setMaximum(100)
    panel.threshold_slider.setValue(
        int(panel.settings["broll"]["matching_threshold"] * 100)
    )
    panel.threshold_slider.valueChanged.connect(panel._on_threshold_changed)
    card_layout.addWidget(panel.threshold_slider)

    card_layout.addSpacing(6)

    button_layout = QHBoxLayout()
    button_layout.setSpacing(8)

    panel.match_btn = QPushButton("Match B-roll")
    panel.match_btn.setCursor(Qt.PointingHandCursor)
    panel.match_btn.clicked.connect(panel._on_match_clicked)

    panel.apply_btn = QPushButton("Apply to Timeline")
    panel.apply_btn.setObjectName("PrimaryButton")
    panel.apply_btn.setCursor(Qt.PointingHandCursor)
    panel.apply_btn.clicked.connect(panel._on_apply_clicked)

    button_layout.addWidget(panel.match_btn, 1)
    button_layout.addWidget(panel.apply_btn, 2)
    card_layout.addLayout(button_layout)

    parent_layout.addWidget(card)
