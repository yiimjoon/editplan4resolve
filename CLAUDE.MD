# editρlan for DaVinci Resolve

> **Technical Name:** EditPlan for Resolve
> **Display Name:** editρlan for Resolve

Plan-driven AI editing plugin for DaVinci Resolve (Transcription + B-roll Matching + Multicam Sync)

---

## 📌 Quick Reference

### Naming Strategy
- **UI/Brand:** editρlan (ρ = Rho, creator signature + engineering precision)
- **System/Package:** EditPlan (file paths, CLI compatibility)

### Stack
Python 3.12 · PySide6 · SQLite (FTS5 + Vector) · OpenCLIP · Whisper · OpenCV · Resolve API

### Execution
- **Launch:** `Workspace → Scripts → Comp → VideoForge`
- **Deploy:** `robocopy VideoForge "$env:APPDATA\...\Scripts\Comp\VideoForge\VideoForge" /E`
- **Logs:** `%APPDATA%\Blackmagic Design\DaVinci Resolve\...\EditPlan.log`

---

## 🏗️ Architecture

```
VideoForge/
├── adapters/      # CLIP, Audio, Video, SAM3/Audio (WSL subprocess)
├── ai/            # LLM hooks, ComfyUI prompt generation
├── broll/         # Indexer, Matcher, Placer, Generators, Quality checker
├── core/          # Transcriber, SegmentStore, TimelineBuilder, SyncMatcher
├── integrations/  # ResolveAPI
├── multicam/      # BoundaryDetector, AngleScorer, AngleSelector, PlanGenerator
├── ui/
│   ├── icons/             # SVG sidebar icons (6)
│   ├── sections/          # Page builders (analyze, edit, library, match, sync, beat, multicam, settings)
│   ├── widgets/           # CollapsibleCard
│   ├── library_manager.py # Standalone DB tool
│   └── main_panel.py      # VSCode-style sidebar
└── config/        # settings.yaml, z_image_turbo.json (ComfyUI)
```

---

## ⚡ Core Rules

1. **Resolve API → Main Thread ONLY** (no Worker threads)
2. **Never delete Whisper model** (del/gc.collect → crash)
3. **Draft Mode is non-destructive** (UID tracking, render-time application)
4. **OpenCV scene detection preferred** (25% accuracy boost over time-based)
5. **Quality validation + retry** (max 3 retries for low-quality stock/AI)
6. **OpenCV subprocess mode** (avoid Resolve DLL conflicts)
7. **SAM models via WSL subprocess** (HuggingFace official environment)

---

## 🎯 Feature Summary

| Phase | Feature | Key Files | Details |
|-------|---------|-----------|---------|
| **3** | ComfyUI Integration | `comfyui_generator.py`, `prompt_generator.py` | AI B-roll generation with LLM prompts |
| **4** | OpenCV Analysis | `video_analyzer.py`, `scene_detector.py`, `quality_checker.py` | Scene cuts (Bhattacharyya), quality gates (blur/brightness/noise) |
| **5** | Library Manager | `library_manager.py` | Standalone DB tool: pagination, thumbnails, metadata edit, batch ops |
| **6** | SAM3 + Audio | `sam3_worker.py`, `sam_audio_worker.py` | Manual video segmentation & audio separation (WSL) |
| **7** | Global/Local Library | `library_section.py` | Dual DB (global shared + project-local), search scope toggle |
| **8** | Audio Sync | `audio_sync.py`, `sync_matcher.py` | Silence pattern matching (same/inverse), cross-correlation, offset calculation |
| **9** | Hybrid Sync | `voice_detector.py`, `content_matcher.py` | Silero VAD (speech), librosa (chroma/onset) |
| **11** | Multicam Cutter | `boundary_detector.py`, `angle_scorer.py`, `plan_generator.py` | Sentence-based cuts, OpenCV angle scoring, rule-based selection |

---

## 🎨 UI (v2.0.0)

### VSCode-Style Sidebar
- **6 pages:** Transcribe, Edit, Multicam, B-roll, Beat, Settings
- **Navigation:** Click icons OR Ctrl+1~6, Ctrl+[ / ]
- **Animations:** 150ms fade transitions, collapsible card slides
- **Status bar:** 3-part (status | timeline info | error count)

### DaVinci Color Palette
```python
bg_window: #1F1F1F   bg_panel: #292929   bg_card: #252525
text_main: #CCCCCC   text_dim: #999999   accent: #4C8BF5
success: #6E9C76     error: #F54E4E      warning: #FFA500
```

---

## ⚙️ Config Keys (Critical)

### Library
```python
global_library_db_path  # Global shared library
local_library_db_path   # Project-local library
library_search_scope    # "both" | "global" | "local"
```

### Audio Sync
```python
audio_sync_threshold_db: -40.0       # Silence threshold
audio_sync_min_silence: 0.3          # Min duration (s)
audio_sync_max_offset: 30.0          # Search window (s)
audio_sync_min_confidence: 0.5       # Match threshold
```

### Multicam
```python
multicam_max_segment_sec: 10.0       # Max segment length
multicam_min_hold_sec: 2.0           # Min hold time
multicam_max_repeat: 3               # Max same-angle repeats
multicam_boundary_mode: "hybrid"     # "sentence" | "fixed" | "hybrid"
```

---

## 📖 Version History

- **v2.0.0** (2026-01-02) - UI Redesign + Rebranding to editρlan
- **v1.12.0** (2025-01-XX) - Phase 11 Auto Multicam Cutter
- **v1.10.0** (2025-XX-XX) - VideoForge legacy

---

## 🔧 Session Handoff Template

```markdown
[Session Context]
Date: YYYY-MM-DD
Branch: [branch-name]
Last Commit: [hash] - [message]

[Goal]
- Primary: [what you're trying to achieve]
- Phase: [if applicable]

[Recent Changes]
- Modified: [file1.py, file2.py]
- Added: [file3.py]
- Behavior: [what changed]

[Known Issues]
- [Bug/Warning description]
- Main-thread violations: [yes/no]

[Tool Routing]
- UI action → Handler: [main thread]
- Heavy work → Worker: [background thread]

[LLM Config]
- Provider: [gemini/openai/disabled]
- Model: [model-name]
- API Key: [configured/missing]

[Deploy]
robocopy VideoForge "$env:APPDATA\...\VideoForge\VideoForge" /E
robocopy scripts\Comp\VideoForge "$env:APPDATA\...\VideoForge" /E

[Verification]
- Launch: Workspace → Scripts → Comp → VideoForge
- Check: [UI indicator / log line]
```

**Critical Reminders:**
- Resolve API calls MUST be on main thread
- UI updates MUST be on main thread
- Worker threads MUST NOT call Resolve API
- Never fabricate results—return status-only if uncertain
