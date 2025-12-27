"""Misc tools section builder."""

from __future__ import annotations

from VideoForge.config.config_manager import Config
from VideoForge.ui.qt_compat import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    Qt,
)


def build_misc_section(panel, parent_layout: QVBoxLayout) -> None:
    """Build the Misc tools tab."""
    card = panel._create_card()
    card_layout = QVBoxLayout(card)
    card_layout.setSpacing(8)

    card_layout.addWidget(panel._create_section_title("5. Misc Tools"))

    columns = QHBoxLayout()
    columns.setSpacing(12)
    left_col = QVBoxLayout()
    right_col = QVBoxLayout()
    columns.addLayout(left_col, 1)
    columns.addLayout(right_col, 1)
    card_layout.addLayout(columns)

    # --- Column 1: SAM3 & SAM Audio Tools ---
    left_col.addWidget(panel._create_section_title("A. SAM3 Tools"))

    sam3_note = QLabel("Segment video by prompt (manual tool; no pipeline usage).")
    sam3_note.setStyleSheet(f"color: {panel.colors['text_dim']}; font-size: 10px;")
    left_col.addWidget(sam3_note)

    sam3_prompt_label = QLabel("Prompt:")
    sam3_prompt_label.setStyleSheet(f"color: {panel.colors['text_dim']}; font-size: 11px;")
    left_col.addWidget(sam3_prompt_label)

    panel.sam3_prompt_edit = QLineEdit()
    panel.sam3_prompt_edit.setPlaceholderText("e.g. person, car, object")
    panel.sam3_prompt_edit.setText(str(Config.get("sam3_prompt") or ""))
    panel.sam3_prompt_edit.textChanged.connect(panel._on_sam3_prompt_changed)
    left_col.addWidget(panel.sam3_prompt_edit)

    sam3_input_label = QLabel("Input Video:")
    sam3_input_label.setStyleSheet(f"color: {panel.colors['text_dim']}; font-size: 11px;")
    left_col.addWidget(sam3_input_label)

    panel.sam3_input_edit = QLineEdit()
    panel.sam3_input_edit.setPlaceholderText("Select video file...")
    panel.sam3_input_edit.setText(str(Config.get("sam3_input_path") or ""))
    panel.sam3_input_edit.textChanged.connect(panel._on_sam3_input_changed)
    panel.sam3_input_browse_btn = QPushButton("Browse")
    panel.sam3_input_browse_btn.setFixedWidth(60)
    panel.sam3_input_browse_btn.clicked.connect(panel._on_sam3_input_browse)
    sam3_input_row = QHBoxLayout()
    sam3_input_row.addWidget(panel.sam3_input_edit)
    sam3_input_row.addWidget(panel.sam3_input_browse_btn)
    left_col.addLayout(sam3_input_row)

    sam3_output_label = QLabel("Output Folder (optional):")
    sam3_output_label.setStyleSheet(f"color: {panel.colors['text_dim']}; font-size: 11px;")
    left_col.addWidget(sam3_output_label)

    panel.sam3_output_edit = QLineEdit()
    panel.sam3_output_edit.setPlaceholderText("Use input folder if empty")
    panel.sam3_output_edit.setText(str(Config.get("sam3_output_dir") or ""))
    panel.sam3_output_edit.textChanged.connect(panel._on_sam3_output_changed)
    panel.sam3_output_browse_btn = QPushButton("Browse")
    panel.sam3_output_browse_btn.setFixedWidth(60)
    panel.sam3_output_browse_btn.clicked.connect(panel._on_sam3_output_browse)
    sam3_output_row = QHBoxLayout()
    sam3_output_row.addWidget(panel.sam3_output_edit)
    sam3_output_row.addWidget(panel.sam3_output_browse_btn)
    left_col.addLayout(sam3_output_row)

    sam3_size_label = QLabel("SAM3 Model Size:")
    sam3_size_label.setStyleSheet(f"color: {panel.colors['text_dim']}; font-size: 11px;")
    left_col.addWidget(sam3_size_label)

    panel.sam3_model_combo = QComboBox()
    panel.sam3_model_combo.addItems(["small", "base", "large"])
    panel.sam3_model_combo.setCurrentText(str(Config.get("sam3_model_size", "large")))
    panel.sam3_model_combo.currentTextChanged.connect(panel._on_sam3_model_changed)
    left_col.addWidget(panel.sam3_model_combo)

    panel.sam3_run_btn = QPushButton("Run SAM3 Segmentation")
    panel.sam3_run_btn.setCursor(Qt.PointingHandCursor)
    panel.sam3_run_btn.clicked.connect(panel._on_sam3_run_clicked)
    left_col.addWidget(panel.sam3_run_btn)

    left_col.addSpacing(12)
    left_col.addWidget(panel._create_section_title("B. SAM Audio Tools"))

    sam_audio_note = QLabel("Isolate audio by prompt (manual tool).")
    sam_audio_note.setStyleSheet(f"color: {panel.colors['text_dim']}; font-size: 10px;")
    left_col.addWidget(sam_audio_note)

    prompt_label = QLabel("Isolation Prompt:")
    prompt_label.setStyleSheet(f"color: {panel.colors['text_dim']}; font-size: 11px;")
    left_col.addWidget(prompt_label)

    panel.sam_audio_prompt_edit = QLineEdit()
    panel.sam_audio_prompt_edit.setPlaceholderText("e.g. voice, applause, music")
    panel.sam_audio_prompt_edit.setText(str(Config.get("sam_audio_prompt") or ""))
    panel.sam_audio_prompt_edit.textChanged.connect(panel._on_sam_audio_prompt_changed)
    left_col.addWidget(panel.sam_audio_prompt_edit)

    input_label = QLabel("Input File:")
    input_label.setStyleSheet(f"color: {panel.colors['text_dim']}; font-size: 11px;")
    left_col.addWidget(input_label)

    panel.sam_audio_input_edit = QLineEdit()
    panel.sam_audio_input_edit.setPlaceholderText("Select audio/video file...")
    panel.sam_audio_input_edit.setText(str(Config.get("sam_audio_input_path") or ""))
    panel.sam_audio_input_edit.textChanged.connect(panel._on_sam_audio_input_changed)
    panel.sam_audio_input_browse_btn = QPushButton("Browse")
    panel.sam_audio_input_browse_btn.setFixedWidth(60)
    panel.sam_audio_input_browse_btn.clicked.connect(panel._on_sam_audio_input_browse)
    sam_audio_input_row = QHBoxLayout()
    sam_audio_input_row.addWidget(panel.sam_audio_input_edit)
    sam_audio_input_row.addWidget(panel.sam_audio_input_browse_btn)
    left_col.addLayout(sam_audio_input_row)

    output_label = QLabel("Output Folder (optional):")
    output_label.setStyleSheet(f"color: {panel.colors['text_dim']}; font-size: 11px;")
    left_col.addWidget(output_label)

    panel.sam_audio_output_edit = QLineEdit()
    panel.sam_audio_output_edit.setPlaceholderText("Use input folder if empty")
    panel.sam_audio_output_edit.setText(str(Config.get("sam_audio_output_dir") or ""))
    panel.sam_audio_output_edit.textChanged.connect(panel._on_sam_audio_output_changed)
    panel.sam_audio_output_browse_btn = QPushButton("Browse")
    panel.sam_audio_output_browse_btn.setFixedWidth(60)
    panel.sam_audio_output_browse_btn.clicked.connect(panel._on_sam_audio_output_browse)
    sam_audio_output_row = QHBoxLayout()
    sam_audio_output_row.addWidget(panel.sam_audio_output_edit)
    sam_audio_output_row.addWidget(panel.sam_audio_output_browse_btn)
    left_col.addLayout(sam_audio_output_row)

    sam_audio_size_label = QLabel("SAM Audio Model Size:")
    sam_audio_size_label.setStyleSheet(f"color: {panel.colors['text_dim']}; font-size: 11px;")
    left_col.addWidget(sam_audio_size_label)

    panel.sam_audio_model_combo = QComboBox()
    panel.sam_audio_model_combo.addItems(["base", "large"])
    panel.sam_audio_model_combo.setCurrentText(str(Config.get("sam_audio_model_size", "large")))
    panel.sam_audio_model_combo.currentTextChanged.connect(panel._on_sam_audio_model_changed)
    left_col.addWidget(panel.sam_audio_model_combo)

    panel.sam_audio_run_btn = QPushButton("Extract SAM Audio")
    panel.sam_audio_run_btn.setCursor(Qt.PointingHandCursor)
    panel.sam_audio_run_btn.clicked.connect(panel._on_sam_audio_run_clicked)
    left_col.addWidget(panel.sam_audio_run_btn)

    left_col.addSpacing(12)
    left_col.addWidget(panel._create_section_title("C. WSL Environment"))

    wsl_note = QLabel("Used by SAM3/SAM Audio subprocess tools.")
    wsl_note.setStyleSheet(f"color: {panel.colors['text_dim']}; font-size: 10px;")
    left_col.addWidget(wsl_note)

    distro_row = QHBoxLayout()
    distro_row.addWidget(QLabel("Distro:"))
    panel.wsl_distro_input = QLineEdit()
    panel.wsl_distro_input.setText(Config.get("sam3_wsl_distro", "Ubuntu"))
    panel.wsl_distro_input.setPlaceholderText("Ubuntu")
    panel.wsl_distro_input.textChanged.connect(panel._on_wsl_distro_changed)
    distro_row.addWidget(panel.wsl_distro_input)
    left_col.addLayout(distro_row)

    python_row = QHBoxLayout()
    python_row.addWidget(QLabel("Python:"))
    panel.wsl_python_input = QLineEdit()
    panel.wsl_python_input.setText(Config.get("sam3_wsl_python", "python3"))
    panel.wsl_python_input.setPlaceholderText("python3")
    panel.wsl_python_input.textChanged.connect(panel._on_wsl_python_changed)
    python_row.addWidget(panel.wsl_python_input)
    left_col.addLayout(python_row)

    venv_row = QHBoxLayout()
    venv_row.addWidget(QLabel("venv:"))
    panel.wsl_venv_input = QLineEdit()
    panel.wsl_venv_input.setText(Config.get("sam3_wsl_venv", ""))
    panel.wsl_venv_input.setPlaceholderText("/home/user/vf-sam3")
    panel.wsl_venv_input.textChanged.connect(panel._on_wsl_venv_changed)
    venv_row.addWidget(panel.wsl_venv_input)
    left_col.addLayout(venv_row)

    # --- Column 2: Miscellaneous tools ---
    right_col.addWidget(panel._create_section_title("D. Miscellaneous"))

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

    parent_layout.addWidget(card)
