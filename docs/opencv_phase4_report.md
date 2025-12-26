# Phase 4 OpenCV Report

## Summary
Phase 4 integrates OpenCV-based analysis to improve sampling and quality checks.

## Phase 4-A
- Added OpenCV dependency in `pyproject.toml`.
- Implemented `VideoAnalyzer` for basic video info and frame extraction.
- Added basic tests and integration doc.

## Phase 4-B
- Implemented `SceneDetector` (histogram-based).
- Added scene detection sampling option in `broll/indexer.py`.
- Added Settings controls: scene detection toggle + sensitivity slider.

## Phase 4-C
- Implemented `VideoQualityChecker` (blur/brightness/noise heuristics).
- Added stock quality gating in `DirectBrollGenerator._generate_stock`.
- Added Settings controls: quality check toggle + min quality slider.

## Phase 4-D
- Added OpenCV integration tests.

## Phase 4-E
- Added subprocess execution for OpenCV inside Resolve host.
- Resolve forces subprocess mode to avoid DLL import hangs.
- Subprocess worker tested via `tests/test_opencv_worker_subprocess.py`.

## Config Keys
- `use_scene_detection` (bool)
- `scene_detection_threshold` (int, 10-50)
- `enable_quality_check` (bool)
- `min_quality_score` (float, 0.4-0.9)
- `opencv_subprocess_mode` (bool, Resolve forced True)
- `opencv_python_exe` (optional python.exe path for worker)

## Notes
Tests skip when OpenCV or sample media is unavailable.
