"""Match section builder."""

from __future__ import annotations

from VideoForge.config.config_manager import Config
from VideoForge.ui.qt_compat import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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

    panel.generate_broll_btn = QPushButton("Generate B-roll")
    panel.generate_broll_btn.setCursor(Qt.PointingHandCursor)
    panel.generate_broll_btn.clicked.connect(panel._on_generate_broll_clicked)
    panel.generate_broll_btn.setToolTip("Generate B-roll (after editing transcript)")

    panel.match_btn = QPushButton("Match B-roll")
    panel.match_btn.setCursor(Qt.PointingHandCursor)
    panel.match_btn.clicked.connect(panel._on_match_clicked)

    panel.apply_btn = QPushButton("Apply to Timeline")
    panel.apply_btn.setObjectName("PrimaryButton")
    panel.apply_btn.setCursor(Qt.PointingHandCursor)
    panel.apply_btn.clicked.connect(panel._on_apply_clicked)

    button_layout.addWidget(panel.generate_broll_btn, 1)
    button_layout.addWidget(panel.match_btn, 1)
    button_layout.addWidget(panel.apply_btn, 2)
    card_layout.addLayout(button_layout)

    parent_layout.addWidget(card)

    broll_card = panel._create_card()
    broll_layout = QVBoxLayout(broll_card)
    broll_layout.setSpacing(8)

    broll_layout.addWidget(panel._create_section_title("B-roll Generation"))

    columns = QHBoxLayout()
    columns.setSpacing(12)
    left_col = QVBoxLayout()
    right_col = QVBoxLayout()
    columns.addLayout(left_col, 1)
    columns.addLayout(right_col, 1)
    broll_layout.addLayout(columns)

    def add_setting_row(layout, label_text, widget, label_width=120):
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setFixedWidth(label_width)
        lbl.setStyleSheet("font-family: 'Space Mono', monospace; font-size: 10px; text-transform: lowercase;")
        row.addWidget(lbl)
        row.addWidget(widget)
        layout.addLayout(row)

    panel.pexels_key_edit = QLineEdit()
    panel.pexels_key_edit.setEchoMode(QLineEdit.Password)
    panel.pexels_key_edit.setPlaceholderText("Pexels API key")
    panel.pexels_key_edit.setText(str(Config.get("pexels_api_key") or ""))
    panel.pexels_key_edit.textChanged.connect(panel._on_pexels_key_changed)
    add_setting_row(left_col, "Pexels API", panel.pexels_key_edit)

    panel.unsplash_key_edit = QLineEdit()
    panel.unsplash_key_edit.setEchoMode(QLineEdit.Password)
    panel.unsplash_key_edit.setPlaceholderText("Unsplash API key")
    panel.unsplash_key_edit.setText(str(Config.get("unsplash_api_key") or ""))
    panel.unsplash_key_edit.textChanged.connect(panel._on_unsplash_key_changed)
    add_setting_row(left_col, "Unsplash API", panel.unsplash_key_edit)

    panel.comfyui_url_edit = QLineEdit()
    panel.comfyui_url_edit.setPlaceholderText("http://127.0.0.1:8188")
    panel.comfyui_url_edit.setText(str(Config.get("comfyui_url") or "http://127.0.0.1:8188"))
    panel.comfyui_url_edit.textChanged.connect(panel._on_comfyui_url_changed)
    add_setting_row(left_col, "ComfyUI URL", panel.comfyui_url_edit)

    right_col.addSpacing(4)

    panel.broll_enabled_checkbox = QCheckBox("Enable B-roll generation")
    panel.broll_enabled_checkbox.setChecked(
        bool(Config.get("broll_gen_enabled", Config.get("uvm_enabled", True)))
    )
    panel.broll_enabled_checkbox.stateChanged.connect(panel._on_broll_gen_enabled_changed)
    right_col.addWidget(panel.broll_enabled_checkbox)

    panel.broll_output_edit = QLineEdit()
    panel.broll_output_edit.setPlaceholderText("Output root (generated_broll)")
    default_output = panel._resolve_repo_root() / "generated_broll"
    panel.broll_output_edit.setText(
        str(Config.get("broll_gen_output_dir") or Config.get("uvm_output_dir") or default_output)
    )
    panel.broll_output_edit.textChanged.connect(panel._on_broll_output_changed)
    panel.broll_output_browse_btn = QPushButton("Browse")
    panel.broll_output_browse_btn.setFixedWidth(60)
    panel.broll_output_browse_btn.clicked.connect(panel._on_broll_output_browse)
    broll_output_row = QHBoxLayout()
    broll_output_row.addWidget(panel.broll_output_edit)
    broll_output_row.addWidget(panel.broll_output_browse_btn)
    right_col.addLayout(broll_output_row)

    panel.broll_mode_combo = QComboBox()
    panel.broll_mode_combo.addItems(["stock", "ai", "hybrid"])
    panel.broll_mode_combo.setCurrentText(
        str(Config.get("broll_gen_mode") or Config.get("uvm_mode") or "stock")
    )
    panel.broll_mode_combo.currentTextChanged.connect(panel._on_broll_gen_mode_changed)
    add_setting_row(right_col, "Generation Mode", panel.broll_mode_combo)

    panel.broll_min_scenes_edit = QLineEdit()
    panel.broll_min_scenes_edit.setPlaceholderText("Min")
    min_scenes = Config.get("broll_min_scenes")
    if min_scenes:
        panel.broll_min_scenes_edit.setText(str(min_scenes))
    panel.broll_min_scenes_edit.textChanged.connect(panel._on_broll_min_scenes_changed)
    add_setting_row(right_col, "Min Scenes (LLM)", panel.broll_min_scenes_edit)

    panel.broll_max_scenes_edit = QLineEdit()
    panel.broll_max_scenes_edit.setPlaceholderText("Auto")
    max_scenes = Config.get("broll_max_scenes")
    if max_scenes is None:
        max_scenes = Config.get("broll_gen_max_scenes")
    if max_scenes:
        panel.broll_max_scenes_edit.setText(str(max_scenes))
    panel.broll_max_scenes_edit.textChanged.connect(panel._on_broll_gen_max_changed)
    add_setting_row(right_col, "Max Scenes", panel.broll_max_scenes_edit)

    parent_layout.addWidget(broll_card)
