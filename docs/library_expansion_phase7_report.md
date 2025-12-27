# Phase 7 Library Expansion Report

## Summary
Phase 7 introduces separate Global and Local B-roll libraries with a selectable search scope.

## Concepts
- Global Library: shared stock clips used across projects.
- Local Library: project-specific captures tied to a single project.

## Config Changes
- global_library_db_path: path to shared library DB.
- local_library_db_path: path to project-local library DB.
- library_search_scope: "global" | "local" | "both".

## Migration
- Legacy library_db_path auto-migrates to global_library_db_path.
- local_library_db_path defaults to a project-local name when possible.

## UI Updates
- Settings: F. Library Paths includes Global/Local DB inputs and search scope.
- Main panel: browse + scope change handlers save config immediately.
- Library tab: Scan Global Library / Scan Local Library buttons.
- Library Manager: switch Global/Local via menu + Library Type display in stats.

## Core Logic
- LibraryDB loads both DBs and filters search by scope.
- LibraryDB.upsert_clip supports target_db routing.
- LibraryIndexer indexes global/local libraries via library_type.
- BrollMatcher initializes LibraryDB from Config.

## Usage
1. Set Global/Local DB paths in Settings.
2. Choose Search Scope (Both/Global only/Local only).
3. Scan each library from the Library tab.

## Files Modified
- VideoForge/config/config_manager.py
- VideoForge/broll/db.py
- VideoForge/broll/indexer.py
- VideoForge/broll/matcher.py
- VideoForge/ui/sections/settings_section.py
- scripts/Comp/VideoForge/ui/main_panel.py
- VideoForge/ui/sections/library_section.py
- docs/library_expansion_phase7_report.md
- CLAUDE.MD

## Testing Checklist
- [x] Syntax check (py compileall) for broll modules
- [ ] UI smoke test in Resolve (Settings + Library tab)
- [ ] Library scanning (global/local)
- [ ] Matching with search scope (global/local/both)
