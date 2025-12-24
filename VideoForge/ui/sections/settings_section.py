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
    columns.setSpacing(16)
    left_col = QVBoxLayout()
    right_col = QVBoxLayout()
    columns.addLayout(left_col, 1)
    columns.addLayout(right_col, 1)
    card_layout.addLayout(columns)

    left_col.addWidget(QLabel("Transcription Engine"))
    panel.engine_combo = QComboBox()
    panel.engine_combo.addItems(["Resolve AI", "Whisper"])
    saved_engine = Config.get("transcription_engine", "Whisper")
    if saved_engine in {"Resolve AI", "Whisper"}:
        panel.engine_combo.setCurrentText(saved_engine)
    panel.engine_combo.currentTextChanged.connect(panel._on_engine_changed)
    left_col.addWidget(panel.engine_combo)

    left_col.addWidget(QLabel("Whisper Language"))
    panel.language_combo = QComboBox()
    panel.language_combo.addItems(["auto", "ko", "en"])
    current_lang = panel.settings.get("whisper", {}).get("language", "auto")
    if current_lang in {"auto", "ko", "en"}:
        panel.language_combo.setCurrentText(current_lang)
    else:
        panel.language_combo.setCurrentText("auto")
    panel.language_combo.currentTextChanged.connect(panel._on_language_changed)
    left_col.addWidget(panel.language_combo)

    left_col.addWidget(QLabel("Whisper Task"))
    panel.task_combo = QComboBox()
    panel.task_combo.addItems(["transcribe", "translate"])
    current_task = panel.settings.get("whisper", {}).get("task", "transcribe")
    if current_task in {"transcribe", "translate"}:
        panel.task_combo.setCurrentText(current_task)
    else:
        panel.task_combo.setCurrentText("transcribe")
    panel.task_combo.currentTextChanged.connect(panel._on_task_changed)
    left_col.addWidget(panel.task_combo)

    panel.transcript_chars_label = QLabel("Transcript max chars per line")
    left_col.addWidget(panel.transcript_chars_label)
    panel.transcript_chars_edit = QLineEdit()
    panel.transcript_chars_edit.setFixedWidth(80)
    panel.transcript_chars_edit.setText(str(Config.get("subtitle_max_chars", 42)))
    panel.transcript_chars_edit.textChanged.connect(panel._on_transcript_chars_changed)
    left_col.addWidget(panel.transcript_chars_edit)

    left_col.addWidget(QLabel("LLM Provider"))
    panel.llm_provider_combo = QComboBox()
    panel.llm_provider_combo.addItems(["disabled", "gemini"])
    saved_provider = Config.get("llm_provider")
    if not saved_provider:
        saved_provider = get_llm_provider() or "disabled"
    saved_provider = saved_provider or "disabled"
    if saved_provider in {"disabled", "gemini"}:
        panel.llm_provider_combo.setCurrentText(saved_provider)
    else:
        panel.llm_provider_combo.setCurrentText("disabled")
    panel.llm_provider_combo.currentTextChanged.connect(panel._on_llm_provider_changed)
    left_col.addWidget(panel.llm_provider_combo)

    left_col.addWidget(QLabel("LLM Model"))
    panel.llm_model_edit = QLineEdit()
    llm_model_value = str(Config.get("llm_model") or get_llm_model())
    if llm_model_value.startswith("models/"):
        llm_model_value = llm_model_value[len("models/") :]
    panel.llm_model_edit.setText(llm_model_value)
    panel.llm_model_edit.textChanged.connect(panel._on_llm_model_changed)
    left_col.addWidget(panel.llm_model_edit)

    left_col.addWidget(QLabel("LLM API Key"))
    panel.llm_key_edit = QLineEdit()
    panel.llm_key_edit.setEchoMode(QLineEdit.Password)
    panel.llm_key_edit.setPlaceholderText("Gemini API key")
    panel.llm_key_edit.setText(str(Config.get("llm_api_key") or (get_llm_api_key() or "")))
    panel.llm_key_edit.textChanged.connect(panel._on_llm_key_changed)
    left_col.addWidget(panel.llm_key_edit)

    panel.llm_status_label = QLabel("LLM Status: Inactive")
    panel.llm_status_label.setStyleSheet(
        f"color: {panel.colors['text_dim']}; background: transparent;"
    )
    left_col.addWidget(panel.llm_status_label)

    panel.llm_hint_label = QLabel("Stored in AppData/VideoForge (user_config.json, llm.env).")
    panel.llm_hint_label.setStyleSheet(
        f"color: {panel.colors['text_dim']}; background: transparent;"
    )
    left_col.addWidget(panel.llm_hint_label)

    llm_enabled = panel.llm_provider_combo.currentText() != "disabled"
    panel.llm_model_edit.setEnabled(llm_enabled)
    panel.llm_key_edit.setEnabled(llm_enabled)

    left_col.addWidget(QLabel("LLM Instructions Preset"))
    panel.llm_instructions_mode_combo = QComboBox()
    panel.llm_instructions_mode_combo.addItems(["shortform", "longform"])
    saved_mode = str(Config.get("llm_instructions_mode") or "shortform").strip().lower()
    if saved_mode in {"shortform", "longform"}:
        panel.llm_instructions_mode_combo.setCurrentText(saved_mode)
    else:
        panel.llm_instructions_mode_combo.setCurrentText("shortform")
    panel.llm_instructions_mode_combo.currentTextChanged.connect(
        panel._on_llm_instructions_mode_changed
    )
    left_col.addWidget(panel.llm_instructions_mode_combo)

    srt_row = QHBoxLayout()
    srt_row.setSpacing(6)
    panel.srt_path_edit = QLineEdit()
    panel.srt_path_edit.setPlaceholderText("Optional SRT path...")
    panel.srt_browse_btn = QPushButton("Browse")
    panel.srt_browse_btn.setFixedWidth(72)
    panel.srt_browse_btn.setCursor(Qt.PointingHandCursor)
    panel.srt_browse_btn.clicked.connect(panel._on_browse_srt)
    srt_row.addWidget(panel.srt_path_edit)
    srt_row.addWidget(panel.srt_browse_btn)
    left_col.addLayout(srt_row)

    right_col.addWidget(QLabel("Silence Removal"))
    right_col.addWidget(QLabel("Silence Preset"))
    panel.silence_preset_combo = QComboBox()
    panel.silence_preset_combo.addItems(["Balanced", "Aggressive", "Conservative", "Custom"])
    panel.silence_preset_combo.setCurrentText(str(Config.get("silence_preset") or "Balanced"))
    panel.silence_preset_combo.currentTextChanged.connect(panel._on_silence_preset_changed)
    right_col.addWidget(panel.silence_preset_combo)

    panel.render_min_duration_label = QLabel("Min Segment Duration (sec)")
    right_col.addWidget(panel.render_min_duration_label)
    panel.render_min_duration_edit = QLineEdit()
    panel.render_min_duration_edit.setFixedWidth(80)
    panel.render_min_duration_edit.setText(
        str(panel.settings["silence"].get("render_min_duration", 0.6))
    )
    panel.render_min_duration_edit.textChanged.connect(panel._on_render_min_duration_changed)
    right_col.addWidget(panel.render_min_duration_edit)

    panel.tail_min_duration_label = QLabel("Tail Fragment Min Duration (sec)")
    right_col.addWidget(panel.tail_min_duration_label)
    panel.tail_min_duration_edit = QLineEdit()
    panel.tail_min_duration_edit.setFixedWidth(80)
    panel.tail_min_duration_edit.setText(
        str(panel.settings["silence"].get("tail_min_duration", 0.8))
    )
    panel.tail_min_duration_edit.textChanged.connect(panel._on_tail_min_duration_changed)
    right_col.addWidget(panel.tail_min_duration_edit)

    panel.tail_ratio_label = QLabel("Tail Ratio Threshold (0-1)")
    right_col.addWidget(panel.tail_ratio_label)
    panel.tail_ratio_edit = QLineEdit()
    panel.tail_ratio_edit.setFixedWidth(80)
    panel.tail_ratio_edit.setText(str(panel.settings["silence"].get("tail_ratio", 0.2)))
    panel.tail_ratio_edit.textChanged.connect(panel._on_tail_ratio_changed)
    right_col.addWidget(panel.tail_ratio_edit)

    right_col.addWidget(QLabel("Render Mode"))
    panel.render_mode_combo = QComboBox()
    panel.render_mode_combo.addItems(["Silence Removal", "J/L-cut"])
    saved_render_mode = Config.get("render_mode", "Silence Removal")
    if saved_render_mode not in {"Silence Removal", "J/L-cut"}:
        saved_render_mode = "Silence Removal"
    panel.render_mode_combo.setCurrentText(saved_render_mode)
    panel.render_mode_combo.currentTextChanged.connect(panel._on_render_mode_changed)
    right_col.addWidget(panel.render_mode_combo)

    try:
        panel._on_silence_preset_changed(panel.silence_preset_combo.currentText())
    except Exception:
        pass

    panel.silence_export_btn = QPushButton("Render File (ffmpeg)")
    panel.silence_export_btn.setCursor(Qt.PointingHandCursor)
    panel.silence_export_btn.clicked.connect(panel._on_render_silence_removed_clicked)
    right_col.addWidget(panel.silence_export_btn)

    right_col.addWidget(QLabel("Rendered Output Action"))
    panel.silence_action_combo = QComboBox()
    panel.silence_action_combo.addItems(
        [
            "Manual (Media Pool only)",
            "Insert Above Track (keep original)",
        ]
    )
    saved_action = Config.get("silence_action", "Insert Above Track (keep original)")
    if saved_action in {
        "Manual (Media Pool only)",
        "Insert Above Track (keep original)",
    }:
        panel.silence_action_combo.setCurrentText(saved_action)
    panel.silence_action_combo.currentTextChanged.connect(panel._on_silence_action_changed)
    right_col.addWidget(panel.silence_action_combo)

    panel.silence_replace_btn = QPushButton("Insert/Replace Using Rendered File (debug)")
    panel.silence_replace_btn.setCursor(Qt.PointingHandCursor)
    panel.silence_replace_btn.clicked.connect(panel._on_replace_with_rendered_clicked)
    right_col.addWidget(panel.silence_replace_btn)

    right_col.addWidget(QLabel("J/L-cut Mode"))
    panel.jlcut_mode_combo = QComboBox()
    panel.jlcut_mode_combo.addItems(["Off", "J-cut", "L-cut", "Both", "Auto"])
    saved_jlcut = str(panel.settings["broll"].get("jlcut_mode", "off")).strip().lower()
    jlcut_map = {
        "off": "Off",
        "jcut": "J-cut",
        "lcut": "L-cut",
        "both": "Both",
        "auto": "Auto",
    }
    panel.jlcut_mode_combo.setCurrentText(jlcut_map.get(saved_jlcut, "Off"))
    panel.jlcut_mode_combo.currentTextChanged.connect(panel._on_jlcut_mode_changed)
    right_col.addWidget(panel.jlcut_mode_combo)

    panel.jcut_offset_label = QLabel("J-cut offset (sec)")
    right_col.addWidget(panel.jcut_offset_label)
    panel.jcut_offset_edit = QLineEdit()
    panel.jcut_offset_edit.setFixedWidth(80)
    panel.jcut_offset_edit.setText(str(panel.settings["broll"].get("jcut_offset", 0.0)))
    panel.jcut_offset_edit.textChanged.connect(panel._on_jcut_offset_changed)
    right_col.addWidget(panel.jcut_offset_edit)

    panel.lcut_offset_label = QLabel("L-cut offset (sec)")
    right_col.addWidget(panel.lcut_offset_label)
    panel.lcut_offset_edit = QLineEdit()
    panel.lcut_offset_edit.setFixedWidth(80)
    panel.lcut_offset_edit.setText(str(panel.settings["broll"].get("lcut_offset", 0.0)))
    panel.lcut_offset_edit.textChanged.connect(panel._on_lcut_offset_changed)
    right_col.addWidget(panel.lcut_offset_edit)

    panel.jlcut_overlap_label = QLabel("J/L overlap (frames)")
    right_col.addWidget(panel.jlcut_overlap_label)
    panel.jlcut_overlap_edit = QLineEdit()
    panel.jlcut_overlap_edit.setFixedWidth(80)
    panel.jlcut_overlap_edit.setText(
        str(
            panel.settings["broll"].get(
                "jlcut_overlap_frames",
                Config.get("jlcut_overlap_frames", 15),
            )
        )
    )
    panel.jlcut_overlap_edit.textChanged.connect(panel._on_jlcut_overlap_changed)
    right_col.addWidget(panel.jlcut_overlap_edit)

    cleanup_row = QHBoxLayout()
    cleanup_row.addWidget(QLabel("Project Cleanup"))
    cleanup_row.addStretch()
    panel.cleanup_btn = QPushButton("Clean Old Projects")
    panel.cleanup_btn.setCursor(Qt.PointingHandCursor)
    panel.cleanup_btn.clicked.connect(panel._on_cleanup_projects)
    cleanup_row.addWidget(panel.cleanup_btn)
    right_col.addLayout(cleanup_row)

    panel._update_transcript_option_visibility(panel.engine_combo.currentText())

    right_col.addWidget(QLabel("Hybrid Weights"))
    vector_weight = float(panel.settings["broll"]["hybrid_weights"]["vector"])
    text_weight = float(panel.settings["broll"]["hybrid_weights"]["text"])
    panel.vector_weight_label = QLabel(
        f"Vector: {vector_weight:.2f} / Text: {text_weight:.2f}"
    )
    panel.vector_weight_label.setStyleSheet(
        f"color: {panel.colors['text_dim']}; background: transparent;"
    )
    right_col.addWidget(panel.vector_weight_label)

    panel.vector_weight_slider = QSlider(Qt.Horizontal)
    panel.vector_weight_slider.setMinimum(0)
    panel.vector_weight_slider.setMaximum(100)
    panel.vector_weight_slider.setValue(
        int(panel.settings["broll"]["hybrid_weights"]["vector"] * 100)
    )
    panel.vector_weight_slider.valueChanged.connect(panel._on_vector_weight_changed)
    right_col.addWidget(panel.vector_weight_slider)

    prior_weight = float(panel.settings["broll"]["hybrid_weights"]["prior"])
    panel.prior_weight_label = QLabel(f"Prior Weight: {prior_weight:.2f}")
    panel.prior_weight_label.setStyleSheet(
        f"color: {panel.colors['text_dim']}; background: transparent;"
    )
    right_col.addWidget(panel.prior_weight_label)

    panel.prior_weight_slider = QSlider(Qt.Horizontal)
    panel.prior_weight_slider.setMinimum(0)
    panel.prior_weight_slider.setMaximum(30)
    panel.prior_weight_slider.setValue(
        int(panel.settings["broll"]["hybrid_weights"]["prior"] * 100)
    )
    panel.prior_weight_slider.valueChanged.connect(panel._on_prior_weight_changed)
    right_col.addWidget(panel.prior_weight_slider)

    visual_threshold = float(panel.settings["broll"]["visual_similarity_threshold"])
    panel.visual_threshold_label = QLabel(f"Visual Filter: {visual_threshold:.2f}")
    panel.visual_threshold_label.setStyleSheet(
        f"color: {panel.colors['text_dim']}; background: transparent;"
    )
    right_col.addWidget(panel.visual_threshold_label)

    panel.visual_threshold_slider = QSlider(Qt.Horizontal)
    panel.visual_threshold_slider.setMinimum(70)
    panel.visual_threshold_slider.setMaximum(95)
    panel.visual_threshold_slider.setValue(
        int(panel.settings["broll"]["visual_similarity_threshold"] * 100)
    )
    panel.visual_threshold_slider.valueChanged.connect(panel._on_visual_threshold_changed)
    right_col.addWidget(panel.visual_threshold_slider)

    panel.sidecar_btn = QPushButton("Generate Scene Sidecars")
    panel.sidecar_btn.setCursor(Qt.PointingHandCursor)
    panel.sidecar_btn.clicked.connect(panel._on_generate_sidecars)
    right_col.addWidget(panel.sidecar_btn)

    panel.comfyui_sidecar_btn = QPushButton("Generate ComfyUI Sidecars")
    panel.comfyui_sidecar_btn.setCursor(Qt.PointingHandCursor)
    panel.comfyui_sidecar_btn.clicked.connect(panel._on_generate_comfyui_sidecars)
    right_col.addWidget(panel.comfyui_sidecar_btn)

    parent_layout.addWidget(card)
