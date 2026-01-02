# editρlan for DaVinci Resolve

> **Plan-driven precision editing for professionals**

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)]()
[![Resolve](https://img.shields.io/badge/DaVinci%20Resolve-18.5%2B-orange.svg)]()
[![Python](https://img.shields.io/badge/python-3.12-green.svg)]()

---

## 🎯 What is editρlan?

**editρlan** (pronounced "Edit-Rho-Plan") is a plan-review-apply workflow automation plugin for DaVinci Resolve, designed by **Rho Young Woo** for professional video editors.

### Why ρ (Rho)?

- **Creator Signature:** Named after the developer's last name (Rho)
- **Engineering Symbolism:**
  - **ρ = Density** (Physics): Tight, precise edits with no waste
  - **ρ = Correlation** (Statistics): Perfect sync/match of timeline elements
- **Professional Tool Aesthetic:** A specialist tool, not a consumer app

---

## ✨ Core Philosophy: Plan → Review → Apply

Unlike "auto" tools, editρlan emphasizes **human control**:

1. **Plan:** Define your editing strategy (transcription, B-roll matching, multicam sync)
2. **Review:** Approve AI-generated plans before execution
3. **Apply:** Precise, non-destructive timeline updates

---

## 🚀 Quick Start (3 Steps)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Deploy to Resolve

Note: The technical package name remains `VideoForge` for compatibility. The UI displays **editρlan**.

```bash
# Windows
robocopy VideoForge "%APPDATA%\Blackmagic Design\DaVinci Resolve\Support\Fusion\Scripts\Comp\VideoForge" /E

# macOS/Linux
cp -r VideoForge ~/Library/Application\ Support/Blackmagic\ Design/DaVinci\ Resolve/Fusion/Scripts/Comp/VideoForge
```

### 3. Launch in Resolve

Workspace → Scripts → Comp → VideoForge → VideoForge

---

## 📚 Documentation

- User Guide
- Developer Guide
- API Reference

## 🎨 UI Preview

VSCode-style sidebar with DaVinci Resolve color palette.

## 🛠️ Tech Stack

- UI: PySide6, VSCode-style navigation
- AI: OpenCLIP (ViT-B-32), Faster-Whisper (large-v3)
- Video: OpenCV (scene detection, quality check)
- Backend: SQLite (FTS5 + Vector), DaVinci Resolve Scripting API

## 📖 Version History

- v2.0.0 (2026-01-02): Complete UI redesign + Rebranding to editρlan
- v1.10.0 (2025-XX-XX): Previous version (VideoForge legacy)

## 📄 License

MIT License - See LICENSE for details

## 👤 Author

Rho Young Woo

Specialist video editor & automation engineer

editρlan - Precision editing, planned.
