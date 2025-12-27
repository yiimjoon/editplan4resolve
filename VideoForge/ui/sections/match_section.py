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

    scope_row = QHBoxLayout()
    scope_label = QLabel("Library Scope")
    scope_label.setStyleSheet(f"color: {panel.colors['text_dim']}; font-size: 11px;")
    panel.match_scope_combo = QComboBox()
    panel.match_scope_combo.addItems(
        ["Both (Global + Local)", "Global only", "Local only"]
    )
    current_scope = Config.get("library_search_scope", "both")
    scope_map = {"both": 0, "global": 1, "local": 2}
    panel.match_scope_combo.setCurrentIndex(scope_map.get(current_scope, 0))
    panel.match_scope_combo.currentIndexChanged.connect(panel._on_library_scope_changed)
    scope_row.addWidget(scope_label)
    scope_row.addWidget(panel.match_scope_combo)
    card_layout.addLayout(scope_row)

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

    align_row = QHBoxLayout()
    align_label = QLabel("Alignment")
    align_label.setStyleSheet(f"color: {panel.colors['text_dim']}; font-size: 11px;")
    panel.alignment_status_label = QLabel("Alignment: -")
    panel.alignment_status_label.setStyleSheet(
        f"color: {panel.colors['text_dim']}; font-size: 11px;"
    )
    panel.reset_alignment_btn = QPushButton("Reset Alignment")
    panel.reset_alignment_btn.setCursor(Qt.PointingHandCursor)
    panel.reset_alignment_btn.clicked.connect(panel._on_reset_alignment_clicked)
    align_row.addWidget(align_label)
    align_row.addWidget(panel.alignment_status_label)
    align_row.addStretch()
    align_row.addWidget(panel.reset_alignment_btn)
    card_layout.addLayout(align_row)

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

    panel.match_source_combo = QComboBox()
    panel.match_source_combo.addItems(
        [
            "Library Only",
            "Library + Stock",
            "Library + AI",
            "Library + Stock + AI (Full)",
            "Stock Only",
            "AI Only",
        ]
    )
    saved_source = Config.get("broll_source_mode", "stock_only")
    source_map = {
        "library_only": "Library Only",
        "library_stock": "Library + Stock",
        "library_ai": "Library + AI",
        "library_stock_ai": "Library + Stock + AI (Full)",
        "stock_only": "Stock Only",
        "ai_only": "AI Only",
    }
    panel.match_source_combo.setCurrentText(source_map.get(saved_source, "Stock Only"))
    panel.match_source_combo.currentTextChanged.connect(panel._on_broll_source_changed)
    add_setting_row(right_col, "B-roll Source", panel.match_source_combo)

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

    panel.reset_broll_btn = QPushButton("Reset Project B-roll")
    panel.reset_broll_btn.setCursor(Qt.PointingHandCursor)
    panel.reset_broll_btn.clicked.connect(panel._on_reset_project_broll_clicked)
    right_col.addWidget(panel.reset_broll_btn)

    parent_layout.addWidget(broll_card)
