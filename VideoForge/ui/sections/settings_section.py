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
from VideoForge.ui.widgets.collapsible_card import CollapsibleCard


def build_settings_section(panel, parent_layout: QVBoxLayout) -> None:
    """Build the Advanced settings cards."""
    columns = QHBoxLayout()
    columns.setSpacing(12)
    left_col = QVBoxLayout()
    left_col.setSpacing(12)
    right_col = QVBoxLayout()
    right_col.setSpacing(12)
    columns.addLayout(left_col, 1)
    columns.addLayout(right_col, 1)
    parent_layout.addLayout(columns)

    # --- Helper: Vertical group with horizontal label+widget ---
    def add_setting_row(layout, label_text, widget, label_width=160):
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setFixedWidth(label_width)
        lbl.setStyleSheet("font-family: 'JetBrains Mono', 'Consolas', monospace; font-size: 10px; text-transform: lowercase;")
        row.addWidget(lbl)
        row.addWidget(widget)
        layout.addLayout(row)

    # --- Column 1: Transcription & LLM ---
    intelligence_card = CollapsibleCard(
        "A. Intelligence", section_id="settings_intelligence", collapsed=False
    )
    left_col.addWidget(intelligence_card)
    intelligence_layout = intelligence_card.content_layout
    
    panel.engine_combo = QComboBox()
    panel.engine_combo.addItems(["Resolve AI", "Whisper"])
    saved_engine = Config.get("transcription_engine", "Whisper")
    if saved_engine in {"Resolve AI", "Whisper"}:
        panel.engine_combo.setCurrentText(saved_engine)
    panel.engine_combo.currentTextChanged.connect(panel._on_engine_changed)
    add_setting_row(intelligence_layout, "Transcription Engine", panel.engine_combo)

    panel.language_combo = QComboBox()
    panel.language_combo.addItems(["auto", "ko", "en"])
    current_lang = panel.settings.get("whisper", {}).get("language", "auto")
    if current_lang in {"auto", "ko", "en"}:
        panel.language_combo.setCurrentText(current_lang)
    else:
        panel.language_combo.setCurrentText("auto")
    panel.language_combo.currentTextChanged.connect(panel._on_language_changed)
    add_setting_row(intelligence_layout, "Whisper Language", panel.language_combo)

    panel.task_combo = QComboBox()
    panel.task_combo.addItems(["transcribe", "translate"])
    current_task = panel.settings.get("whisper", {}).get("task", "transcribe")
    if current_task in {"transcribe", "translate"}:
        panel.task_combo.setCurrentText(current_task)
    else:
        panel.task_combo.setCurrentText("transcribe")
    panel.task_combo.currentTextChanged.connect(panel._on_task_changed)
    add_setting_row(intelligence_layout, "Whisper Task", panel.task_combo)

    panel.transcript_chars_edit = QLineEdit()
    panel.transcript_chars_edit.setFixedWidth(60)
    panel.transcript_chars_edit.setText(str(Config.get("subtitle_max_chars", 42)))
    panel.transcript_chars_edit.textChanged.connect(panel._on_transcript_chars_changed)
    add_setting_row(
        intelligence_layout, "Per-line Chars (Resolve AI)", panel.transcript_chars_edit
    )

    llm_card = CollapsibleCard(
        "B. LLM Configuration", section_id="settings_llm", collapsed=True
    )
    left_col.addWidget(llm_card)
    llm_layout = llm_card.content_layout

    panel.llm_provider_combo = QComboBox()
    panel.llm_provider_combo.addItems(["disabled", "gemini"])
    saved_provider = Config.get("llm_provider") or get_llm_provider() or "disabled"
    if saved_provider in {"disabled", "gemini"}:
        panel.llm_provider_combo.setCurrentText(saved_provider)
    else:
        panel.llm_provider_combo.setCurrentText("disabled")
    panel.llm_provider_combo.currentTextChanged.connect(panel._on_llm_provider_changed)
    add_setting_row(llm_layout, "LLM Provider", panel.llm_provider_combo)

    panel.llm_model_edit = QLineEdit()
    llm_model_value = str(Config.get("llm_model") or get_llm_model())
    if llm_model_value.startswith("models/"):
        llm_model_value = llm_model_value[len("models/") :]
    panel.llm_model_edit.setText(llm_model_value)
    panel.llm_model_edit.textChanged.connect(panel._on_llm_model_changed)
    add_setting_row(llm_layout, "LLM Model", panel.llm_model_edit)

    panel.llm_key_edit = QLineEdit()
    panel.llm_key_edit.setEchoMode(QLineEdit.Password)
    panel.llm_key_edit.setPlaceholderText("Gemini API key")
    panel.llm_key_edit.setText(str(Config.get("llm_api_key") or (get_llm_api_key() or "")))
    panel.llm_key_edit.textChanged.connect(panel._on_llm_key_changed)
    add_setting_row(llm_layout, "LLM API Key", panel.llm_key_edit)

    panel.llm_max_tokens_edit = QLineEdit()
    panel.llm_max_tokens_edit.setFixedWidth(80)
    panel.llm_max_tokens_edit.setText(str(Config.get("llm_max_tokens", 4096)))
    panel.llm_max_tokens_edit.textChanged.connect(panel._on_llm_max_tokens_changed)
    add_setting_row(llm_layout, "LLM Max Tokens", panel.llm_max_tokens_edit)

    panel.llm_instructions_mode_combo = QComboBox()
    panel.llm_instructions_mode_combo.addItems(["shortform", "longform"])
    saved_mode = str(Config.get("llm_instructions_mode") or "shortform").strip().lower()
    panel.llm_instructions_mode_combo.setCurrentText(saved_mode if saved_mode in {"shortform", "longform"} else "shortform")
    panel.llm_instructions_mode_combo.currentTextChanged.connect(panel._on_llm_instructions_mode_changed)
    add_setting_row(llm_layout, "Instructions Preset", panel.llm_instructions_mode_combo)

    panel.llm_status_label = QLabel("LLM Status: Inactive")
    panel.llm_status_label.setStyleSheet(f"color: {panel.colors['text_dim']}; font-size: 11px;")
    llm_layout.addWidget(panel.llm_status_label)

    # --- Column 2: B-roll & Library ---
    broll_card = CollapsibleCard(
        "D. B-roll Generation",
        section_id="settings_broll_generation",
        collapsed=True,
    )
    right_col.addWidget(broll_card)
    broll_layout = broll_card.content_layout

    panel.quality_check_checkbox = QCheckBox("Enable Quality Check (Stock/AI)")
    quality_check_value = Config.get("enable_quality_check")
    panel.quality_check_checkbox.setChecked(
        True if quality_check_value is None else bool(quality_check_value)
    )
    panel.quality_check_checkbox.stateChanged.connect(panel._on_quality_check_changed)
    broll_layout.addWidget(panel.quality_check_checkbox)

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
    broll_layout.addWidget(panel.min_quality_label)

    panel.min_quality_slider = QSlider(Qt.Horizontal)
    panel.min_quality_slider.setMinimum(30)
    panel.min_quality_slider.setMaximum(90)
    panel.min_quality_slider.setValue(int(min_quality_value))
    panel.min_quality_slider.valueChanged.connect(panel._on_min_quality_changed)
    broll_layout.addWidget(panel.min_quality_slider)

    library_card = CollapsibleCard(
        "F. Library Paths",
        section_id="settings_library_paths",
        collapsed=True,
    )
    right_col.addWidget(library_card)
    library_layout = library_card.content_layout

    global_label = QLabel("Global Library (shared):")
    global_label.setStyleSheet(f"color: {panel.colors['text_dim']}; font-size: 11px;")
    library_layout.addWidget(global_label)

    global_layout = QHBoxLayout()
    panel.global_library_path_input = QLineEdit()
    panel.global_library_path_input.setText(str(Config.get("global_library_db_path", "")))
    panel.global_library_path_input.setPlaceholderText("E:/BrollLibrary/global.db")
    panel.global_library_path_input.textChanged.connect(panel._on_global_library_path_changed)
    global_layout.addWidget(panel.global_library_path_input)

    panel.global_library_browse_btn = QPushButton("Browse...")
    panel.global_library_browse_btn.clicked.connect(panel._on_browse_global_library)
    global_layout.addWidget(panel.global_library_browse_btn)

    panel.global_library_unlock_btn = QPushButton("Change")
    panel.global_library_unlock_btn.clicked.connect(panel._on_unlock_global_library)
    global_layout.addWidget(panel.global_library_unlock_btn)
    library_layout.addLayout(global_layout)

    local_label = QLabel("Local Library (project-specific):")
    local_label.setStyleSheet(f"color: {panel.colors['text_dim']}; font-size: 11px;")
    library_layout.addWidget(local_label)

    local_layout = QHBoxLayout()
    panel.local_library_path_input = QLineEdit()
    panel.local_library_path_input.setText(str(Config.get("local_library_db_path", "")))
    panel.local_library_path_input.setPlaceholderText("{project_folder}/local_broll.db")
    panel.local_library_path_input.textChanged.connect(panel._on_local_library_path_changed)
    local_layout.addWidget(panel.local_library_path_input)

    local_browse_btn = QPushButton("Browse...")
    local_browse_btn.clicked.connect(panel._on_browse_local_library)
    local_layout.addWidget(local_browse_btn)
    library_layout.addLayout(local_layout)

    scope_label = QLabel("Search Scope:")
    scope_label.setStyleSheet(f"color: {panel.colors['text_dim']}; font-size: 11px;")
    library_layout.addWidget(scope_label)

    panel.library_scope_combo = QComboBox()
    panel.library_scope_combo.addItems(
        ["Both (Global + Local)", "Global only", "Local only"]
    )
    current_scope = str(Config.get("library_search_scope", "both")).strip().lower()
    scope_map = {"both": 0, "global": 1, "local": 2}
    panel.library_scope_combo.setCurrentIndex(scope_map.get(current_scope, 0))
    panel.library_scope_combo.currentIndexChanged.connect(panel._on_library_scope_changed)
    library_layout.addWidget(panel.library_scope_combo)

    panel.open_library_manager_btn = QPushButton("Open Library Manager")
    panel.open_library_manager_btn.setCursor(Qt.PointingHandCursor)
    panel.open_library_manager_btn.clicked.connect(panel._on_open_library_manager)
    library_layout.addWidget(panel.open_library_manager_btn)

    agent_card = CollapsibleCard(
        "G. Agent Settings",
        section_id="settings_agent",
        collapsed=True,
    )
    right_col.addWidget(agent_card)
    agent_layout = agent_card.content_layout

    agent_mode_label = QLabel("Agent Mode:")
    agent_mode_label.setStyleSheet(f"color: {panel.colors['text_dim']}; font-size: 11px;")
    agent_layout.addWidget(agent_mode_label)

    panel.agent_mode_combo = QComboBox()
    panel.agent_mode_combo.addItems(["recommend_only", "approve_required", "full_access"])
    current_mode = str(Config.get("agent_mode", "approve_required")).strip()
    if current_mode in {"recommend_only", "approve_required", "full_access"}:
        panel.agent_mode_combo.setCurrentText(current_mode)
    panel.agent_mode_combo.currentTextChanged.connect(panel._on_agent_mode_changed)
    agent_layout.addWidget(panel.agent_mode_combo)

    agent_key_label = QLabel("Agent API Key (Gemini):")
    agent_key_label.setStyleSheet(f"color: {panel.colors['text_dim']}; font-size: 11px;")
    agent_layout.addWidget(agent_key_label)

    panel.agent_api_key_input = QLineEdit()
    panel.agent_api_key_input.setEchoMode(QLineEdit.Password)
    panel.agent_api_key_input.setPlaceholderText("Gemini API key (uses LLM key if empty)")
    agent_key_value = str(Config.get("agent_api_key", "")).strip()
    if not agent_key_value:
        agent_key_value = str(Config.get("llm_api_key", "")).strip()
    panel.agent_api_key_input.setText(agent_key_value)
    panel.agent_api_key_input.textChanged.connect(panel._on_agent_api_key_changed)
    agent_layout.addWidget(panel.agent_api_key_input)

    panel.agent_test_btn = QPushButton("Test Agent Connection")
    panel.agent_test_btn.clicked.connect(panel._on_agent_test_connection)
    agent_layout.addWidget(panel.agent_test_btn)

    panel.agent_status_label = QLabel("Agent Status: Not connected")
    panel.agent_status_label.setStyleSheet(
        f"color: {panel.colors['text_dim']}; font-size: 11px;"
    )
    if str(Config.get("agent_api_key", "")).strip() or str(Config.get("llm_api_key", "")).strip():
        panel.agent_status_label.setText("Agent Status: Connected")
    agent_layout.addWidget(panel.agent_status_label)

    hybrid_card = CollapsibleCard(
        "H. Hybrid Search Weights",
        section_id="settings_hybrid_weights",
        collapsed=True,
    )
    right_col.addWidget(hybrid_card)
    hybrid_layout = hybrid_card.content_layout

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

    add_weight_slider(
        hybrid_layout,
        "vector_weight_label",
        "vector_weight_slider",
        "Vector vs Text Weight",
        0,
        100,
        panel.settings["broll"]["hybrid_weights"]["vector"] * 100,
        panel._on_vector_weight_changed,
    )
    
    add_weight_slider(
        hybrid_layout,
        "prior_weight_label",
        "prior_weight_slider",
        "Prior Weight",
        0,
        30,
        panel.settings["broll"]["hybrid_weights"]["prior"] * 100,
        panel._on_prior_weight_changed,
    )

    add_weight_slider(
        hybrid_layout,
        "visual_threshold_label",
        "visual_threshold_slider",
        "Visual Threshold",
        70,
        95,
        panel.settings["broll"]["visual_similarity_threshold"] * 100,
        panel._on_visual_threshold_changed,
    )

    multicam_card = CollapsibleCard(
        "I. Multicam Settings",
        section_id="settings_multicam",
        collapsed=True,
    )
    right_col.addWidget(multicam_card)
    multicam_layout = multicam_card.content_layout

    panel.face_model_dir_input = QLineEdit()
    panel.face_model_dir_input.setPlaceholderText("Optional face model directory")
    panel.face_model_dir_input.setText(str(Config.get("multicam_face_model_dir", "")))
    panel.face_model_dir_input.textChanged.connect(
        lambda value: Config.set("multicam_face_model_dir", value)
    )
    add_setting_row(
        multicam_layout, "Face Model Dir (Optional)", panel.face_model_dir_input
    )

    panel.multicam_edl_dir_input = QLineEdit()
    panel.multicam_edl_dir_input.setPlaceholderText("Optional EDL output dir or .edl path")
    panel.multicam_edl_dir_input.setText(str(Config.get("multicam_edl_output_dir", "")))
    panel.multicam_edl_dir_input.textChanged.connect(
        lambda value: Config.set("multicam_edl_output_dir", value)
    )
    add_setting_row(
        multicam_layout, "EDL Output Path (Optional)", panel.multicam_edl_dir_input
    )

    left_col.addStretch()
    right_col.addStretch()

    # Final logic
    panel._update_transcript_option_visibility(panel.engine_combo.currentText())
    llm_enabled = panel.llm_provider_combo.currentText() != "disabled"
    panel.llm_model_edit.setEnabled(llm_enabled)
    panel.llm_key_edit.setEnabled(llm_enabled)
    panel.llm_max_tokens_edit.setEnabled(llm_enabled)

    agent_enabled = llm_enabled
    panel.agent_mode_combo.setEnabled(agent_enabled)
    panel.agent_api_key_input.setEnabled(agent_enabled)
    panel.agent_test_btn.setEnabled(agent_enabled)
    if not agent_enabled:
        panel.agent_status_label.setText("Agent Status: Disabled (LLM off)")

    global_path = str(Config.get("global_library_db_path", "") or "").strip()
    panel._set_global_library_locked(bool(global_path))

