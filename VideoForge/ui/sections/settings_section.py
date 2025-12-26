"""Settings section builder."""

from __future__ import annotations

from VideoForge.config.config_manager import Config
from VideoForge.ai.llm_hooks import get_llm_api_key, get_llm_model, get_llm_provider
from VideoForge.ui.qt_compat import (
    QComboBox,
    QHBoxLayout,
    QCheckBox,
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
    add_setting_row(mid_col, "J/L jitter (+/-)", panel.jlcut_overlap_jitter_edit)

    mid_col.addSpacing(12)
    mid_col.addWidget(panel._create_section_title("D. B-roll Generation"))

    panel.broll_source_combo = QComboBox()
    panel.broll_source_combo.addItems(
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
    panel.broll_source_combo.setCurrentText(source_map.get(saved_source, "Stock Only"))
    panel.broll_source_combo.currentTextChanged.connect(panel._on_broll_source_changed)
    add_setting_row(mid_col, "B-roll Source", panel.broll_source_combo, label_width=120)

    panel.broll_media_type_combo = QComboBox()
    panel.broll_media_type_combo.addItems(
        ["Auto (Source Default)", "Prefer Video", "Images Only"]
    )
    saved_media = Config.get("broll_prefer_media_type", "auto")
    media_map = {
        "auto": "Auto (Source Default)",
        "prefer_video": "Prefer Video",
        "image_only": "Images Only",
    }
    panel.broll_media_type_combo.setCurrentText(
        media_map.get(saved_media, "Auto (Source Default)")
    )
    panel.broll_media_type_combo.currentTextChanged.connect(panel._on_broll_media_type_changed)
    add_setting_row(mid_col, "Media Type", panel.broll_media_type_combo, label_width=120)

    panel.broll_scene_seed_edit = QLineEdit()
    panel.broll_scene_seed_edit.setFixedWidth(80)
    panel.broll_scene_seed_edit.setPlaceholderText("Random")
    saved_scene_seed = Config.get("broll_scene_seed")
    panel.broll_scene_seed_edit.setText("" if saved_scene_seed in {None, ""} else str(saved_scene_seed))
    panel.broll_scene_seed_edit.textChanged.connect(panel._on_broll_scene_seed_changed)
    add_setting_row(mid_col, "Scene Seed (opt)", panel.broll_scene_seed_edit, label_width=120)

    panel.comfyui_resolution_combo = QComboBox()
    panel.comfyui_resolution_combo.addItems(
        [
            "1920x1024 (default)",
            "1920x1080 (16:9)",
            "1024x1024 (1:1)",
            "1024x1824 (9:16)",
        ]
    )
    saved_res = str(Config.get("comfyui_resolution") or "1920x1024")
    res_map = {
        "1920x1024": "1920x1024 (default)",
        "1920x1080": "1920x1080 (16:9)",
        "1024x1024": "1024x1024 (1:1)",
        "1024x1824": "1024x1824 (9:16)",
    }
    panel.comfyui_resolution_combo.setCurrentText(
        res_map.get(saved_res, "1920x1024 (default)")
    )
    panel.comfyui_resolution_combo.currentTextChanged.connect(
        panel._on_comfyui_resolution_changed
    )
    add_setting_row(mid_col, "ComfyUI Resolution", panel.comfyui_resolution_combo, label_width=120)

    panel.comfyui_seed_edit = QLineEdit()
    panel.comfyui_seed_edit.setPlaceholderText("Random each run")
    panel.comfyui_seed_edit.setFixedWidth(140)
    panel.comfyui_seed_edit.setText(str(Config.get("comfyui_seed") or ""))
    panel.comfyui_seed_edit.textChanged.connect(panel._on_comfyui_seed_changed)
    add_setting_row(mid_col, "ComfyUI Seed (optional)", panel.comfyui_seed_edit, label_width=120)

    panel.comfyui_output_edit = QLineEdit()
    panel.comfyui_output_edit.setPlaceholderText("C:/ComfyUI/ComfyUI/output")
    panel.comfyui_output_edit.setText(str(Config.get("comfyui_output_dir") or ""))
    panel.comfyui_output_edit.textChanged.connect(panel._on_comfyui_output_changed)
    panel.comfyui_output_browse_btn = QPushButton("Browse")
    panel.comfyui_output_browse_btn.setFixedWidth(60)
    panel.comfyui_output_browse_btn.clicked.connect(panel._on_comfyui_output_browse)
    comfyui_output_row = QHBoxLayout()
    comfyui_output_row.addWidget(panel.comfyui_output_edit)
    comfyui_output_row.addWidget(panel.comfyui_output_browse_btn)
    output_label = QLabel("ComfyUI Output Folder (optional)")
    output_label.setStyleSheet("font-size: 10px; color: rgba(26, 26, 26, 0.6);")
    mid_col.addWidget(output_label)
    mid_col.addLayout(comfyui_output_row)

    panel.comfyui_run_script_edit = QLineEdit()
    panel.comfyui_run_script_edit.setPlaceholderText("C:/ComfyUI/ComfyUI/run_comfyui.bat")
    panel.comfyui_run_script_edit.setText(str(Config.get("comfyui_run_script") or ""))
    panel.comfyui_run_script_edit.textChanged.connect(panel._on_comfyui_run_script_changed)
    panel.comfyui_run_script_browse_btn = QPushButton("Browse")
    panel.comfyui_run_script_browse_btn.setFixedWidth(60)
    panel.comfyui_run_script_browse_btn.clicked.connect(panel._on_comfyui_run_script_browse)
    comfyui_run_row = QHBoxLayout()
    comfyui_run_row.addWidget(panel.comfyui_run_script_edit)
    comfyui_run_row.addWidget(panel.comfyui_run_script_browse_btn)
    run_label = QLabel("ComfyUI Run Script (optional)")
    run_label.setStyleSheet("font-size: 10px; color: rgba(26, 26, 26, 0.6);")
    mid_col.addWidget(run_label)
    mid_col.addLayout(comfyui_run_row)

    panel.comfyui_autostart_checkbox = QCheckBox("Auto-start ComfyUI server")
    autostart_value = Config.get("comfyui_autostart")
    panel.comfyui_autostart_checkbox.setChecked(
        True if autostart_value is None else bool(autostart_value)
    )
    panel.comfyui_autostart_checkbox.stateChanged.connect(
        panel._on_comfyui_autostart_changed
    )
    mid_col.addWidget(panel.comfyui_autostart_checkbox)

    # --- Column 3: Matching & Sidecars ---
    right_col.addWidget(panel._create_section_title("E. Video Analysis"))

    panel.scene_detection_checkbox = QCheckBox("Use Scene Detection (OpenCV)")
    scene_detection_value = Config.get("use_scene_detection")
    panel.scene_detection_checkbox.setChecked(
        True if scene_detection_value is None else bool(scene_detection_value)
    )
    panel.scene_detection_checkbox.stateChanged.connect(panel._on_scene_detection_changed)
    right_col.addWidget(panel.scene_detection_checkbox)

    raw_threshold = Config.get("scene_detection_threshold")
    try:
        threshold_value = int(float(raw_threshold)) if raw_threshold is not None else 30
    except (TypeError, ValueError):
        threshold_value = 30

    panel.scene_threshold_label = QLabel(f"Sensitivity: {threshold_value}")
    panel.scene_threshold_label.setStyleSheet(
        f"color: {panel.colors['text_dim']}; font-size: 11px;"
    )
    right_col.addWidget(panel.scene_threshold_label)

    panel.scene_threshold_slider = QSlider(Qt.Horizontal)
    panel.scene_threshold_slider.setMinimum(10)
    panel.scene_threshold_slider.setMaximum(50)
    panel.scene_threshold_slider.setValue(int(threshold_value))
    panel.scene_threshold_slider.valueChanged.connect(panel._on_scene_threshold_changed)
    right_col.addWidget(panel.scene_threshold_slider)

    panel.quality_check_checkbox = QCheckBox("Enable Quality Check (Stock/AI)")
    quality_check_value = Config.get("enable_quality_check")
    panel.quality_check_checkbox.setChecked(
        True if quality_check_value is None else bool(quality_check_value)
    )
    panel.quality_check_checkbox.stateChanged.connect(panel._on_quality_check_changed)
    right_col.addWidget(panel.quality_check_checkbox)

    raw_min_quality = Config.get("min_quality_score")
    try:
        min_quality_value = int(float(raw_min_quality) * 100) if raw_min_quality is not None else 60
    except (TypeError, ValueError):
        min_quality_value = 60
    min_quality_value = max(30, min(90, min_quality_value))

    panel.min_quality_label = QLabel(f"Min Quality: {min_quality_value}%")
    panel.min_quality_label.setStyleSheet(
        f"color: {panel.colors['text_dim']}; font-size: 11px;"
    )
    right_col.addWidget(panel.min_quality_label)

    panel.min_quality_slider = QSlider(Qt.Horizontal)
    panel.min_quality_slider.setMinimum(30)
    panel.min_quality_slider.setMaximum(90)
    panel.min_quality_slider.setValue(int(min_quality_value))
    panel.min_quality_slider.valueChanged.connect(panel._on_min_quality_changed)
    right_col.addWidget(panel.min_quality_slider)

    right_col.addSpacing(12)
    right_col.addWidget(panel._create_section_title("F. Library"))

    panel.library_path_display = QLineEdit()
    panel.library_path_display.setReadOnly(True)
    panel.library_path_display.setPlaceholderText("Library DB path...")
    if hasattr(panel, "library_path_edit"):
        panel.library_path_display.setText(panel.library_path_edit.text().strip())
    add_setting_row(right_col, "Current DB", panel.library_path_display, label_width=120)

    panel.open_library_manager_btn = QPushButton("Open Library Manager")
    panel.open_library_manager_btn.setCursor(Qt.PointingHandCursor)
    panel.open_library_manager_btn.clicked.connect(panel._on_open_library_manager)
    right_col.addWidget(panel.open_library_manager_btn)

    right_col.addSpacing(12)
    right_col.addWidget(panel._create_section_title("G. Advanced AI (SAM Models)"))

    panel.sam3_checkbox = QCheckBox("Enable SAM3 Auto-Tagging (WSL)")
    panel.sam3_checkbox.setChecked(Config.get("use_sam3_tagging", False))
    panel.sam3_checkbox.setToolTip(
        "Automatically detect objects in B-roll videos using SAM3\n"
        "Requires WSL environment with SAM3 installed"
    )
    panel.sam3_checkbox.stateChanged.connect(panel._on_sam3_tagging_changed)
    right_col.addWidget(panel.sam3_checkbox)

    sam3_size_label = QLabel("SAM3 Model Size:")
    sam3_size_label.setStyleSheet(
        f"color: {panel.colors['text_dim']}; font-size: 11px;"
    )
    right_col.addWidget(sam3_size_label)

    panel.sam3_model_combo = QComboBox()
    panel.sam3_model_combo.addItems(["small", "base", "large"])
    panel.sam3_model_combo.setCurrentText(
        str(Config.get("sam3_model_size", "large"))
    )
    panel.sam3_model_combo.currentTextChanged.connect(panel._on_sam3_model_changed)
    right_col.addWidget(panel.sam3_model_combo)

    right_col.addSpacing(8)

    panel.sam_audio_checkbox = QCheckBox("Enable SAM Audio Preprocessing (WSL)")
    panel.sam_audio_checkbox.setChecked(
        Config.get("use_sam_audio_preprocessing", False)
    )
    panel.sam_audio_checkbox.setToolTip(
        "Remove background noise before Whisper transcription\n"
        "Improves accuracy for noisy audio"
    )
    panel.sam_audio_checkbox.stateChanged.connect(
        panel._on_sam_audio_preprocessing_changed
    )
    right_col.addWidget(panel.sam_audio_checkbox)

    sam_audio_size_label = QLabel("SAM Audio Model Size:")
    sam_audio_size_label.setStyleSheet(
        f"color: {panel.colors['text_dim']}; font-size: 11px;"
    )
    right_col.addWidget(sam_audio_size_label)

    panel.sam_audio_model_combo = QComboBox()
    panel.sam_audio_model_combo.addItems(["base", "large"])
    panel.sam_audio_model_combo.setCurrentText(
        str(Config.get("sam_audio_model_size", "large"))
    )
    panel.sam_audio_model_combo.currentTextChanged.connect(
        panel._on_sam_audio_model_changed
    )
    right_col.addWidget(panel.sam_audio_model_combo)

    right_col.addSpacing(12)

    wsl_label = QLabel("WSL Environment:")
    wsl_label.setStyleSheet(
        f"color: {panel.colors['text_dim']}; font-size: 11px; font-weight: bold;"
    )
    right_col.addWidget(wsl_label)

    distro_layout = QHBoxLayout()
    distro_layout.addWidget(QLabel("Distro:"))
    panel.wsl_distro_input = QLineEdit()
    panel.wsl_distro_input.setText(Config.get("sam3_wsl_distro", "Ubuntu"))
    panel.wsl_distro_input.setPlaceholderText("Ubuntu")
    panel.wsl_distro_input.textChanged.connect(panel._on_wsl_distro_changed)
    distro_layout.addWidget(panel.wsl_distro_input)
    right_col.addLayout(distro_layout)

    python_layout = QHBoxLayout()
    python_layout.addWidget(QLabel("Python:"))
    panel.wsl_python_input = QLineEdit()
    panel.wsl_python_input.setText(Config.get("sam3_wsl_python", "python3"))
    panel.wsl_python_input.setPlaceholderText("python3")
    panel.wsl_python_input.textChanged.connect(panel._on_wsl_python_changed)
    python_layout.addWidget(panel.wsl_python_input)
    right_col.addLayout(python_layout)

    venv_layout = QHBoxLayout()
    venv_layout.addWidget(QLabel("venv:"))
    panel.wsl_venv_input = QLineEdit()
    panel.wsl_venv_input.setText(Config.get("sam3_wsl_venv", ""))
    panel.wsl_venv_input.setPlaceholderText("/home/user/vf-sam3")
    panel.wsl_venv_input.textChanged.connect(panel._on_wsl_venv_changed)
    venv_layout.addWidget(panel.wsl_venv_input)
    right_col.addLayout(venv_layout)

    right_col.addSpacing(12)
    right_col.addWidget(panel._create_section_title("H. Hybrid Search Weights"))

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
    right_col.addWidget(panel._create_section_title("I. Maintenance"))

    # --- Maintenance & Misc ---
    right_col.addSpacing(12)
    right_col.addWidget(panel._create_section_title("J. Miscellaneous"))
    
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
