# Phase 6 Report: SAM3 + SAM Audio (WSL Subprocess)

## Summary
Phase 6 integrates SAM3 (video segmentation) and SAM Audio (speech separation)
via WSL subprocesses to avoid Windows DLL conflicts.

## Phase 6-A: Workers + Subprocess Bridges
Files:
- `VideoForge/adapters/sam3_worker.py`
- `VideoForge/adapters/sam3_subprocess.py`
- `VideoForge/adapters/sam3_adapter.py`
- `VideoForge/adapters/sam_audio_worker.py`
- `VideoForge/adapters/sam_audio_subprocess.py`
- `VideoForge/adapters/sam_audio_adapter.py`

Key behavior:
- Windows calls WSL with `wsl -d Ubuntu bash -lc ...`
- WSL venv activation supported (`sam3_wsl_venv`)
- JSON payload returned to Windows

## Phase 6-B: Core Integration
### SAM3 auto-tagging
File: `VideoForge/broll/indexer.py`
- Controlled by `use_sam3_tagging`
- Tags merged into existing `vision_tags`

### SAM Audio preprocessing
File: `VideoForge/core/transcriber.py`
- Controlled by `use_sam_audio_preprocessing`
- Extracts WAV, runs SAM Audio, then Whisper
- Fallback to original audio on failure

## Phase 6-C: Settings UI
File: `VideoForge/ui/sections/settings_section.py`
- G. Advanced AI (SAM Models)
- Toggles + model size + WSL config inputs

File: `scripts/Comp/VideoForge/ui/main_panel.py`
- 7 handlers to persist settings and log changes

## Config Keys
```python
# SAM3
Config.get("use_sam3_tagging", False)
Config.get("sam3_model_size", "large")
Config.get("sam3_wsl_distro", "Ubuntu")
Config.get("sam3_wsl_python", "python3")
Config.get("sam3_wsl_venv", "/home/<user>/vf-sam3")

# SAM Audio
Config.get("use_sam_audio_preprocessing", False)
Config.get("sam_audio_model_size", "large")
Config.get("sam_audio_wsl_distro", "Ubuntu")
Config.get("sam_audio_wsl_python", "python3")
Config.get("sam_audio_wsl_venv", "/home/<user>/vf-sam3")
```

## Usage Examples
### SAM3 tagging during library scan
Enable:
- Settings → G. Advanced AI → Enable SAM3 Auto-Tagging

Run:
- Analyze tab → Scan library

### SAM Audio preprocessing for Whisper
Enable:
- Settings → G. Advanced AI → Enable SAM Audio Preprocessing

Run:
- Analyze tab → Analyze (Whisper)

## Notes
- WSL setup guide: `docs/sam_wsl_setup.md`
- WSL model access must be approved on HuggingFace

