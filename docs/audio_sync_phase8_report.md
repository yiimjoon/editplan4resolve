# Phase 8 Audio-based Sync Report

## Summary
Phase 8 introduces automatic video synchronization based on audio silence pattern
matching, enabling multi-camera and A-roll/B-roll alignment without manual
timecode adjustment.

## Use Cases

### Case 1: Multi-camera Sync (Same Pattern)
Scenario: multiple cameras capture the same scene with different start times.
- Camera 1 (reference): starts at T=0s
- Camera 2 (target A): starts at T=3s
- Camera 3 (target B): starts at T=1s

Result (24fps):
- Track 1: Camera 1 at frame 0
- Track 2: Camera 2 at frame +72
- Track 3: Camera 3 at frame +24

### Case 2: A-roll/B-roll Sync (Inverse Pattern)
Scenario: face cam (A-roll) and screen recording (B-roll) with opposite audio
patterns.
- A-roll: voice during talking, silence during typing
- B-roll: screen audio during typing, silence during talking

Use "Inverse Pattern" mode to match A-roll sound to B-roll silence.

## Core Components

### Phase 8-A: Silence Detection
File: `VideoForge/adapters/audio_sync.py`

Algorithm:
1. Extract audio via ffmpeg (mono, 16kHz, PCM).
2. Compute RMS per frame (default 50ms).
3. Convert to dB: `20 * log10(RMS + 1e-10)`.
4. Detect silence by threshold (default -40dB).
5. Keep continuous ranges >= min_silence (default 0.3s).

Output: `[(start_time, end_time), ...]` in seconds.

### Phase 8-B: Pattern Matching
File: `VideoForge/core/sync_matcher.py`

Algorithm:
1. Convert silence ranges into a binary pattern (100ms resolution).
2. Mode selection:
   - Same: silence matches silence.
   - Inverse: silence matches sound (bitwise NOT).
3. Cross-correlation (SciPy or NumPy fallback).
4. Find max correlation within +/- max_offset (default 30s).
5. Confidence: `max_corr / min(len(ref), len(tgt))`.

Output:
```python
{
    "offset": -2.3,      # seconds (negative = target is late)
    "confidence": 0.87,  # 0.0-1.0
    "ref_silences": [...],
    "tgt_silences": [...],
}
```

### Phase 8-C: UI Integration
Files:
- `VideoForge/ui/sections/sync_section.py`
- `scripts/Comp/VideoForge/ui/main_panel.py`

Features:
- Reference file picker
- Target list with add/clear
- Same/Inverse mode selection
- Auto Sync button
- Status with confidence warnings

### Phase 8-D: Timeline Placement
File: `VideoForge/integrations/resolve_api.py`

Method:
`add_synced_clips_to_timeline(reference_path, sync_results, start_frame=0)`

Workflow:
1. Import reference + targets to Media Pool.
2. Place reference on Track 1 at start_frame (playhead if 0).
3. For each target:
   - `offset_frames = int(-offset_seconds * fps)`
   - place on Track 2+ at `start_frame + offset_frames`.

Offset sign convention:
- offset < 0: target is late, place after reference.
- offset > 0: target is early, place before reference.

## Config Parameters
Defined in `VideoForge/config/config_manager.py`:
```python
Config.get("audio_sync_threshold_db", -40.0)  # silence threshold (dB)
Config.get("audio_sync_min_silence", 0.3)     # min silence duration (sec)
Config.get("audio_sync_max_offset", 30.0)     # max search window (sec)
Config.get("audio_sync_resolution", 0.1)      # pattern resolution (sec)
Config.get("audio_sync_min_confidence", 0.5)  # minimum confidence
```

## Files Modified

New files:
- `VideoForge/adapters/audio_sync.py`
- `VideoForge/core/sync_matcher.py`
- `VideoForge/ui/sections/sync_section.py`
- `tests/test_sync_matcher.py`
- `docs/audio_sync_phase8_report.md`

Modified files:
- `VideoForge/config/config_manager.py`
- `VideoForge/integrations/resolve_api.py`
- `scripts/Comp/VideoForge/ui/main_panel.py`
- `CLAUDE.MD`

## Testing

Unit tests:
- `tests/test_sync_matcher.py`

Run:
```
python -m pytest tests/test_sync_matcher.py -v
```

Integration checklist:
- Sync tab visible in Resolve.
- Reference/Target selection works.
- Same pattern sync works with multi-camera footage.
- Inverse pattern sync works with A-roll + screen capture.
- Timeline placement uses correct offsets.
- Low confidence warning appears.

## Usage

Workflow:
1. Open Sync tab in Resolve.
2. Select reference (A-roll).
3. Add target files (B-roll/multi-camera).
4. Choose mode:
   - Same Pattern: multi-camera with shared audio.
   - Inverse Pattern: A-roll voice vs B-roll screen audio.
5. Click "Auto Sync & Place in Timeline".

Status:
- Green: placed in timeline.
- Orange: low confidence.
- Red: error.

## Troubleshooting
- Low confidence: check audio quality, try opposite mode, or sync manually.
- No silence detected: lower threshold or choose content with silences.
- Offset reversed: swap reference/target or try inverse mode.

## Performance
Baseline (16kHz audio):
- Silence detection: ~5s for 10 min video.
- Pattern matching: ~0.1s per pair (30s window).

Optimization:
- Downsample audio to 16kHz.
- Limit search window (max_offset).
- Keep resolution at 0.1s.

## Known Limitations
- Requires audio track.
- Continuous audio yields low confidence.
- Assumes a single constant offset (no drift).

## Future Enhancements
- Visual sync fallback (OpenCV).
- Drift correction.
- Progress bar during sync.
- Manual offset adjustment.
- Waveform preview.

## Completion Checklist

### Phase 8-A: Silence Detection
- [x] SilenceDetector implemented
- [x] ffmpeg audio extraction
- [x] RMS/dB computation
- [x] Range detection + filtering

### Phase 8-B: Pattern Matching
- [x] SyncMatcher implemented
- [x] Same/Inverse modes
- [x] Cross-correlation fallback
- [x] Confidence scoring
- [x] Unit tests

### Phase 8-C: UI Integration
- [x] Sync tab created
- [x] Reference/Target pickers
- [x] Auto Sync button
- [x] Status and warnings

### Phase 8-D: Timeline Placement
- [x] Add synced clips to timeline
- [x] Track auto-creation
- [x] Offset conversion
- [x] UI integration

### Phase 8-E: Documentation
- [x] Phase 8 report
- [x] CLAUDE.MD updated
- [x] Version bumped to v1.10.0

## Version
v1.10.0 (2025-12-27) - Phase 8 Audio-based Sync
