"""Edit section builder (silence removal, J/L cut, scene cuts)."""

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


def build_edit_section(panel, parent_layout: QVBoxLayout) -> None:
    """Build the Edit section cards."""

    def add_setting_row(layout, label_text, widget, label_width=160):
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setFixedWidth(label_width)
        lbl.setStyleSheet("font-family: 'JetBrains Mono', 'Consolas', monospace; font-size: 10px; text-transform: lowercase;")
        row.addWidget(lbl)
        row.addWidget(widget)
        layout.addLayout(row)

    # --- Card: Silence Removal + J/L-cut ---
    silence_card = panel._create_card()
    silence_layout = QVBoxLayout(silence_card)
    silence_layout.setSpacing(8)
    silence_layout.addWidget(panel._create_section_title("Edit"))

    panel.silence_preset_combo = QComboBox()
    panel.silence_preset_combo.addItems(["Balanced", "Aggressive", "Conservative", "Custom"])
    panel.silence_preset_combo.setCurrentText(str(Config.get("silence_preset") or "Balanced"))
    panel.silence_preset_combo.currentTextChanged.connect(panel._on_silence_preset_changed)
    add_setting_row(silence_layout, "Silence Preset", panel.silence_preset_combo)

    panel.silence_threshold_edit = QLineEdit()
    panel.silence_threshold_edit.setFixedWidth(60)
    panel.silence_threshold_edit.setText(str(panel.settings["silence"].get("threshold_db", -38.0)))
    panel.silence_threshold_edit.textChanged.connect(panel._on_silence_threshold_changed)
    add_setting_row(silence_layout, "Detect Threshold (dB)", panel.silence_threshold_edit)

    panel.silence_min_keep_edit = QLineEdit()
    panel.silence_min_keep_edit.setFixedWidth(60)
    panel.silence_min_keep_edit.setText(str(panel.settings["silence"].get("min_keep_duration", 0.3)))
    panel.silence_min_keep_edit.textChanged.connect(panel._on_silence_min_keep_changed)
    add_setting_row(silence_layout, "Min Keep (s)", panel.silence_min_keep_edit)

    panel.silence_buffer_before_edit = QLineEdit()
    panel.silence_buffer_before_edit.setFixedWidth(60)
    panel.silence_buffer_before_edit.setText(str(panel.settings["silence"].get("buffer_before", 0.25)))
    panel.silence_buffer_before_edit.textChanged.connect(panel._on_silence_buffer_before_changed)
    add_setting_row(silence_layout, "Pad Before (s)", panel.silence_buffer_before_edit)

    panel.silence_buffer_after_edit = QLineEdit()
    panel.silence_buffer_after_edit.setFixedWidth(60)
    panel.silence_buffer_after_edit.setText(str(panel.settings["silence"].get("buffer_after", 0.35)))
    panel.silence_buffer_after_edit.textChanged.connect(panel._on_silence_buffer_after_changed)
    add_setting_row(silence_layout, "Pad After (s)", panel.silence_buffer_after_edit)

    panel.render_min_duration_edit = QLineEdit()
    panel.render_min_duration_edit.setFixedWidth(60)
    panel.render_min_duration_edit.setText(str(panel.settings["silence"].get("render_min_duration", 0.6)))
    panel.render_min_duration_edit.textChanged.connect(panel._on_render_min_duration_changed)
    add_setting_row(silence_layout, "Min Segment (s)", panel.render_min_duration_edit)

    panel.tail_min_duration_edit = QLineEdit()
    panel.tail_min_duration_edit.setFixedWidth(60)
    panel.tail_min_duration_edit.setText(str(panel.settings["silence"].get("tail_min_duration", 0.8)))
    panel.tail_min_duration_edit.textChanged.connect(panel._on_tail_min_duration_changed)
    add_setting_row(silence_layout, "Tail Min Dur (s)", panel.tail_min_duration_edit)

    panel.tail_ratio_edit = QLineEdit()
    panel.tail_ratio_edit.setFixedWidth(60)
    panel.tail_ratio_edit.setText(str(panel.settings["silence"].get("tail_ratio", 0.2)))
    panel.tail_ratio_edit.textChanged.connect(panel._on_tail_ratio_changed)
    add_setting_row(silence_layout, "Tail Ratio (0-1)", panel.tail_ratio_edit)

    panel.render_mode_combo = QComboBox()
    panel.render_mode_combo.addItems(["Silence Removal", "J/L-cut"])
    panel.render_mode_combo.setCurrentText(Config.get("render_mode", "Silence Removal"))
    panel.render_mode_combo.currentTextChanged.connect(panel._on_render_mode_changed)
    add_setting_row(silence_layout, "Render Mode", panel.render_mode_combo)

    panel.silence_action_combo = QComboBox()
    panel.silence_action_combo.addItems(["Manual (Media Pool only)", "Insert Above (keep original)"])
    panel.silence_action_combo.setCurrentText(Config.get("silence_action", "Insert Above (keep original)"))
    panel.silence_action_combo.currentTextChanged.connect(panel._on_silence_action_changed)
    add_setting_row(silence_layout, "Output Action", panel.silence_action_combo)

    panel.silence_export_btn = QPushButton("Render File (ffmpeg)")
    panel.silence_export_btn.setCursor(Qt.PointingHandCursor)
    panel.silence_export_btn.clicked.connect(panel._on_render_silence_removed_clicked)
    silence_layout.addWidget(panel.silence_export_btn)

    panel.silence_replace_btn = QPushButton("Insert/Replace (debug)")
    panel.silence_replace_btn.setCursor(Qt.PointingHandCursor)
    panel.silence_replace_btn.clicked.connect(panel._on_replace_with_rendered_clicked)
    silence_layout.addWidget(panel.silence_replace_btn)

    panel.jlcut_overlap_edit = QLineEdit()
    panel.jlcut_overlap_edit.setFixedWidth(60)
    panel.jlcut_overlap_edit.setText(
        str(
            panel.settings["broll"].get(
                "jlcut_overlap_frames",
                Config.get("jlcut_overlap_frames", 15),
            )
        )
    )
    panel.jlcut_overlap_edit.textChanged.connect(panel._on_jlcut_overlap_changed)
    add_setting_row(silence_layout, "J/L overlap (frames)", panel.jlcut_overlap_edit)

    panel.jlcut_overlap_jitter_edit = QLineEdit()
    panel.jlcut_overlap_jitter_edit.setFixedWidth(60)
    panel.jlcut_overlap_jitter_edit.setText(
        str(
            panel.settings["broll"].get(
                "jlcut_overlap_jitter",
                Config.get("jlcut_overlap_jitter", 0),
            )
        )
    )
    panel.jlcut_overlap_jitter_edit.textChanged.connect(panel._on_jlcut_overlap_jitter_changed)
    add_setting_row(silence_layout, "J/L jitter (+/-)", panel.jlcut_overlap_jitter_edit)

    parent_layout.addWidget(silence_card)

    # --- Card: Scene Detection ---
    scene_card = panel._create_card()
    scene_layout = QVBoxLayout(scene_card)
    scene_layout.setSpacing(8)
    scene_layout.addWidget(panel._create_section_title("Scene Cuts"))

    panel.scene_detection_checkbox = QCheckBox("Use Scene Detection (OpenCV)")
    scene_detection_value = Config.get("use_scene_detection")
    panel.scene_detection_checkbox.setChecked(
        True if scene_detection_value is None else bool(scene_detection_value)
    )
    panel.scene_detection_checkbox.stateChanged.connect(panel._on_scene_detection_changed)
    scene_layout.addWidget(panel.scene_detection_checkbox)

    raw_threshold = Config.get("scene_detection_threshold")
    try:
        threshold_value = int(float(raw_threshold)) if raw_threshold is not None else 8
    except (TypeError, ValueError):
        threshold_value = 8

    panel.scene_threshold_label = QLabel(f"Sensitivity: {threshold_value}")
    panel.scene_threshold_label.setStyleSheet(
        f"color: {panel.colors['text_dim']}; font-size: 11px;"
    )
    scene_layout.addWidget(panel.scene_threshold_label)

    panel.scene_threshold_slider = QSlider(Qt.Horizontal)
    panel.scene_threshold_slider.setMinimum(5)
    panel.scene_threshold_slider.setMaximum(50)
    panel.scene_threshold_slider.setValue(int(threshold_value))
    panel.scene_threshold_slider.valueChanged.connect(panel._on_scene_threshold_changed)
    scene_layout.addWidget(panel.scene_threshold_slider)

    panel.scene_split_btn = QPushButton("Split Selected Clips (Scene Cuts)")
    panel.scene_split_btn.setCursor(Qt.PointingHandCursor)
    panel.scene_split_btn.clicked.connect(panel._on_scene_split_clicked)
    scene_layout.addWidget(panel.scene_split_btn)

    panel.scene_keep_audio_checkbox = QCheckBox("Keep Audio on Scene Cuts")
    keep_audio_value = Config.get("scene_cut_keep_audio", False)
    panel.scene_keep_audio_checkbox.setChecked(bool(keep_audio_value))
    panel.scene_keep_audio_checkbox.stateChanged.connect(panel._on_scene_keep_audio_changed)
    scene_layout.addWidget(panel.scene_keep_audio_checkbox)

    parent_layout.addWidget(scene_card)
