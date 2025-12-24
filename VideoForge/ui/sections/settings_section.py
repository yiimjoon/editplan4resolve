"""Settings section builder."""

from __future__ import annotations

from VideoForge.config.config_manager import Config
from VideoForge.ai.llm_hooks import get_llm_api_key, get_llm_model, get_llm_provider
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


def build_settings_section(panel, parent_layout: QVBoxLayout) -> None:
    """Build the Advanced settings card."""
    card = panel._create_card()
    card_layout = QVBoxLayout(card)
    card_layout.setSpacing(8)

    card_layout.addWidget(panel._create_section_title("4. Advanced"))

    columns = QHBoxLayout()
    columns.setSpacing(12)
    left_col = QVBoxLayout()
    mid_col = QVBoxLayout()
    right_col = QVBoxLayout()
    columns.addLayout(left_col, 1)
    columns.addLayout(mid_col, 1)
    columns.addLayout(right_col, 1)
    card_layout.addLayout(columns)

    # --- Helper: Vertical group with horizontal label+widget ---
    def add_setting_row(layout, label_text, widget, label_width=160):
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setFixedWidth(label_width)
        lbl.setStyleSheet("font-family: 'Space Mono', monospace; font-size: 10px; text-transform: lowercase;")
        row.addWidget(lbl)
        row.addWidget(widget)
        layout.addLayout(row)

    # --- Column 1: Transcription & LLM ---
    left_col.addWidget(panel._create_section_title("A. Intelligence"))
    
    panel.engine_combo = QComboBox()
    panel.engine_combo.addItems(["Resolve AI", "Whisper"])
    saved_engine = Config.get("transcription_engine", "Whisper")
    if saved_engine in {"Resolve AI", "Whisper"}:
        panel.engine_combo.setCurrentText(saved_engine)
    panel.engine_combo.currentTextChanged.connect(panel._on_engine_changed)
    add_setting_row(left_col, "Transcription Engine", panel.engine_combo)

    panel.language_combo = QComboBox()
    panel.language_combo.addItems(["auto", "ko", "en"])
    current_lang = panel.settings.get("whisper", {}).get("language", "auto")
    if current_lang in {"auto", "ko", "en"}:
        panel.language_combo.setCurrentText(current_lang)
    else:
        panel.language_combo.setCurrentText("auto")
    panel.language_combo.currentTextChanged.connect(panel._on_language_changed)
    add_setting_row(left_col, "Whisper Language", panel.language_combo)

    panel.task_combo = QComboBox()
    panel.task_combo.addItems(["transcribe", "translate"])
    current_task = panel.settings.get("whisper", {}).get("task", "transcribe")
    if current_task in {"transcribe", "translate"}:
        panel.task_combo.setCurrentText(current_task)
    else:
        panel.task_combo.setCurrentText("transcribe")
    panel.task_combo.currentTextChanged.connect(panel._on_task_changed)
    add_setting_row(left_col, "Whisper Task", panel.task_combo)

    panel.transcript_chars_edit = QLineEdit()
    panel.transcript_chars_edit.setFixedWidth(60)
    panel.transcript_chars_edit.setText(str(Config.get("subtitle_max_chars", 42)))
    panel.transcript_chars_edit.textChanged.connect(panel._on_transcript_chars_changed)
    add_setting_row(left_col, "Per-line Chars (Resolve AI)", panel.transcript_chars_edit)

    left_col.addSpacing(12)
    left_col.addWidget(panel._create_section_title("B. LLM Configuration"))

    panel.llm_provider_combo = QComboBox()
    panel.llm_provider_combo.addItems(["disabled", "gemini"])
    saved_provider = Config.get("llm_provider") or get_llm_provider() or "disabled"
    if saved_provider in {"disabled", "gemini"}:
        panel.llm_provider_combo.setCurrentText(saved_provider)
    else:
        panel.llm_provider_combo.setCurrentText("disabled")
    panel.llm_provider_combo.currentTextChanged.connect(panel._on_llm_provider_changed)
    add_setting_row(left_col, "LLM Provider", panel.llm_provider_combo)

    panel.llm_model_edit = QLineEdit()
    llm_model_value = str(Config.get("llm_model") or get_llm_model())
    if llm_model_value.startswith("models/"):
        llm_model_value = llm_model_value[len("models/") :]
    panel.llm_model_edit.setText(llm_model_value)
    panel.llm_model_edit.textChanged.connect(panel._on_llm_model_changed)
    add_setting_row(left_col, "LLM Model", panel.llm_model_edit)

    panel.llm_key_edit = QLineEdit()
    panel.llm_key_edit.setEchoMode(QLineEdit.Password)
    panel.llm_key_edit.setPlaceholderText("Gemini API key")
    panel.llm_key_edit.setText(str(Config.get("llm_api_key") or (get_llm_api_key() or "")))
    panel.llm_key_edit.textChanged.connect(panel._on_llm_key_changed)
    add_setting_row(left_col, "LLM API Key", panel.llm_key_edit)

    panel.llm_instructions_mode_combo = QComboBox()
    panel.llm_instructions_mode_combo.addItems(["shortform", "longform"])
    saved_mode = str(Config.get("llm_instructions_mode") or "shortform").strip().lower()
    panel.llm_instructions_mode_combo.setCurrentText(saved_mode if saved_mode in {"shortform", "longform"} else "shortform")
    panel.llm_instructions_mode_combo.currentTextChanged.connect(panel._on_llm_instructions_mode_changed)
    add_setting_row(left_col, "Instructions Preset", panel.llm_instructions_mode_combo)

    panel.llm_status_label = QLabel("LLM Status: Inactive")
    panel.llm_status_label.setStyleSheet(f"color: {panel.colors['text_dim']}; font-size: 11px;")
    left_col.addWidget(panel.llm_status_label)

    # --- Column 2: Silence & J/L-cut ---
    mid_col.addWidget(panel._create_section_title("C. Silence Removal"))

    panel.silence_preset_combo = QComboBox()
    panel.silence_preset_combo.addItems(["Balanced", "Aggressive", "Conservative", "Custom"])
    panel.silence_preset_combo.setCurrentText(str(Config.get("silence_preset") or "Balanced"))
    panel.silence_preset_combo.currentTextChanged.connect(panel._on_silence_preset_changed)
    add_setting_row(mid_col, "Silence Preset", panel.silence_preset_combo)

    panel.render_min_duration_edit = QLineEdit()
    panel.render_min_duration_edit.setFixedWidth(60)
    panel.render_min_duration_edit.setText(str(panel.settings["silence"].get("render_min_duration", 0.6)))
    panel.render_min_duration_edit.textChanged.connect(panel._on_render_min_duration_changed)
    add_setting_row(mid_col, "Min Segment (s)", panel.render_min_duration_edit)

    panel.tail_min_duration_edit = QLineEdit()
    panel.tail_min_duration_edit.setFixedWidth(60)
    panel.tail_min_duration_edit.setText(str(panel.settings["silence"].get("tail_min_duration", 0.8)))
    panel.tail_min_duration_edit.textChanged.connect(panel._on_tail_min_duration_changed)
    add_setting_row(mid_col, "Tail Min Dur (s)", panel.tail_min_duration_edit)

    panel.tail_ratio_edit = QLineEdit()
    panel.tail_ratio_edit.setFixedWidth(60)
    panel.tail_ratio_edit.setText(str(panel.settings["silence"].get("tail_ratio", 0.2)))
    panel.tail_ratio_edit.textChanged.connect(panel._on_tail_ratio_changed)
    add_setting_row(mid_col, "Tail Ratio (0-1)", panel.tail_ratio_edit)

    panel.render_mode_combo = QComboBox()
    panel.render_mode_combo.addItems(["Silence Removal", "J/L-cut"])
    panel.render_mode_combo.setCurrentText(Config.get("render_mode", "Silence Removal"))
    panel.render_mode_combo.currentTextChanged.connect(panel._on_render_mode_changed)
    add_setting_row(mid_col, "Render Mode", panel.render_mode_combo)

    panel.silence_export_btn = QPushButton("Render File (ffmpeg)")
    panel.silence_export_btn.setCursor(Qt.PointingHandCursor)
    panel.silence_export_btn.clicked.connect(panel._on_render_silence_removed_clicked)
    mid_col.addWidget(panel.silence_export_btn)

    panel.silence_action_combo = QComboBox()
    panel.silence_action_combo.addItems(["Manual (Media Pool only)", "Insert Above (keep original)"])
    panel.silence_action_combo.setCurrentText(Config.get("silence_action", "Insert Above (keep original)"))
    panel.silence_action_combo.currentTextChanged.connect(panel._on_silence_action_changed)
    add_setting_row(mid_col, "Output Action", panel.silence_action_combo)

    panel.silence_replace_btn = QPushButton("Insert/Replace (debug)")
    panel.silence_replace_btn.setCursor(Qt.PointingHandCursor)
    panel.silence_replace_btn.clicked.connect(panel._on_replace_with_rendered_clicked)
    mid_col.addWidget(panel.silence_replace_btn)

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
    add_setting_row(mid_col, "J/L overlap (frames)", panel.jlcut_overlap_edit)

    # --- Column 3: Matching & Sidecars ---
    right_col.addWidget(panel._create_section_title("E. Hybrid Search Weights"))

    def add_weight_slider(layout, label_attr, slider_attr, label_text, min_val, max_val, current_val, callback):
        lbl = QLabel(label_text)
        lbl.setStyleSheet(f"color: {panel.colors['text_dim']}; font-size: 11px;")
        setattr(panel, label_attr, lbl)
        layout.addWidget(lbl)
        
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(int(current_val))
        slider.valueChanged.connect(callback)
        setattr(panel, slider_attr, slider)
        layout.addWidget(slider)

    add_weight_slider(right_col, "vector_weight_label", "vector_weight_slider", "Vector vs Text Weight", 0, 100, 
                      panel.settings["broll"]["hybrid_weights"]["vector"] * 100, panel._on_vector_weight_changed)
    
    add_weight_slider(right_col, "prior_weight_label", "prior_weight_slider", "Prior Weight", 0, 30, 
                      panel.settings["broll"]["hybrid_weights"]["prior"] * 100, panel._on_prior_weight_changed)

    add_weight_slider(right_col, "visual_threshold_label", "visual_threshold_slider", "Visual Threshold", 70, 95, 
                      panel.settings["broll"]["visual_similarity_threshold"] * 100, panel._on_visual_threshold_changed)

    right_col.addSpacing(12)
    right_col.addWidget(panel._create_section_title("F. Maintenance"))

    # --- Maintenance & Misc ---
    right_col.addSpacing(12)
    right_col.addWidget(panel._create_section_title("G. Miscellaneous"))
    
    srt_row = QHBoxLayout()
    panel.srt_path_edit = QLineEdit()
    panel.srt_path_edit.setPlaceholderText("Optional SRT path...")
    panel.srt_browse_btn = QPushButton("Browse")
    panel.srt_browse_btn.setFixedWidth(60)
    panel.srt_browse_btn.clicked.connect(panel._on_browse_srt)
    srt_row.addWidget(panel.srt_path_edit)
    srt_row.addWidget(panel.srt_browse_btn)
    right_col.addLayout(srt_row)

    panel.sidecar_btn = QPushButton("Scene Sidecars")
    panel.sidecar_btn.setCursor(Qt.PointingHandCursor)
    panel.sidecar_btn.clicked.connect(panel._on_generate_sidecars)
    right_col.addWidget(panel.sidecar_btn)

    panel.comfyui_sidecar_btn = QPushButton("ComfyUI Sidecars")
    panel.comfyui_sidecar_btn.setCursor(Qt.PointingHandCursor)
    panel.comfyui_sidecar_btn.clicked.connect(panel._on_generate_comfyui_sidecars)
    right_col.addWidget(panel.comfyui_sidecar_btn)

    panel.cleanup_btn = QPushButton("Clean Old Projects")
    panel.cleanup_btn.setCursor(Qt.PointingHandCursor)
    panel.cleanup_btn.clicked.connect(panel._on_cleanup_projects)
    right_col.addWidget(panel.cleanup_btn)

    # Final logic
    panel._update_transcript_option_visibility(panel.engine_combo.currentText())
    llm_enabled = panel.llm_provider_combo.currentText() != "disabled"
    panel.llm_model_edit.setEnabled(llm_enabled)
    panel.llm_key_edit.setEnabled(llm_enabled)

    parent_layout.addWidget(card)
