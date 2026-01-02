# VideoForge

DaVinci Resolve용 AI 자동 편집 플러그인 (음성 인식 + B-roll 매칭 + 컷 편집)

## 스택
- Python 3.12, PySide6, SQLite (FTS5 + Vector)
- OpenCLIP (ViT-B-32), faster-whisper (large-v3, GPU)
- OpenCV (비디오 분석, 장면 전환 감지, 품질 검증)
- DaVinci Resolve Scripting API
- ComfyUI (AI 이미지/비디오 생성)
- SAM3 + SAM Audio (WSL subprocess)

## 구조
```
VideoForge/
├── adapters/          # CLIP, Audio, Video 어댑터
│   ├── clip_adapter.py
│   ├── audio_adapter.py
│   ├── audio_sync.py         # 무음 구간 감지 (Phase 8-A)
│   ├── video_analyzer.py     # OpenCV 비디오 분석 (Phase 4-A)
│   ├── scene_detector.py     # 장면 전환 감지 (Phase 4-B)
│   ├── sam3_worker.py        # WSL SAM3 worker (Phase 6)
│   ├── sam3_subprocess.py    # SAM3 subprocess bridge (Phase 6)
│   ├── sam3_adapter.py       # SAM3 adapter (Phase 6)
│   ├── sam_audio_worker.py   # WSL SAM Audio worker (Phase 6)
│   ├── sam_audio_subprocess.py # SAM Audio subprocess bridge (Phase 6)
│   └── sam_audio_adapter.py  # SAM Audio adapter (Phase 6)
├── ai/
│   ├── llm_hooks.py          # LLM 프롬프트 생성
│   └── prompt_generator.py   # ComfyUI 프롬프트 생성 (Phase 3-C)
├── broll/
│   ├── indexer.py            # 라이브러리 인덱싱 + 장면 기반 샘플링 (Phase 4-B)
│   ├── matcher.py
│   ├── placer.py             # V4 트랙 배치
│   ├── direct_generator.py   # 다이렉트 B-roll 생성 + 품질 검증 (Phase 4-C)
│   ├── comfyui_generator.py  # ComfyUI 워크플로우 실행 (Phase 3-A/B)
│   ├── external_generator.py # 외부 생성기 추상화 (Phase 3-B)
│   ├── generator_bridge.py   # 생성기 통합 레이어 (Phase 3-B)
│   ├── quality_checker.py    # OpenCV 품질 검증 (Phase 4-C)
│   └── database.py
├── core/              # Transcriber, SegmentStore, TimelineBuilder
│   └── sync_matcher.py       # 패턴 매칭 동기화 (Phase 8-B)
├── integrations/      # ResolveAPI
├── ui/
│   ├── sections/
│   │   ├── settings_section.py  # E. Video Analysis, F. Library (Phase 4-5)
│   │   ├── library_section.py   # Library DB 경로 관리
│   │   ├── match_section.py
│   │   ├── sync_section.py      # Audio Sync UI (Phase 8-C)
│   │   └── misc_section.py      # SAM Tools, WSL 환경, Misc 도구 (Phase 6)
│   ├── library_manager.py       # 독립 실행 가능한 DB 관리 도구 (Phase 5)
│   └── main_panel.py            # Misc 탭 통합 (Phase 6)
└── config/
    └── z_image_turbo.json       # ComfyUI 워크플로우 설정 (Phase 3-A)
```

## 실행
- Resolve: `Workspace → Scripts → Comp → VideoForge`
- 배포: `robocopy VideoForge "$env:APPDATA\...\Scripts\Comp\VideoForge\VideoForge" /E`
- 로그: `%APPDATA%\...\VideoForge.log`

## 핵심 규칙
1. **Resolve API는 Main Thread에서만** (Worker 사용 금지)
2. **Whisper 모델 절대 삭제 금지** (del/gc.collect → 크래시)
3. **Draft Mode는 비파괴** (UID 추적, 렌더 시점 반영)
4. **OpenCV 장면 감지 우선** (시간 기반 샘플링보다 25% 정확도 향상)
5. **품질 검증 + 재시도** (저품질 스톡/AI 자동 필터링, 최대 3회 재시도)
6. **OpenCV는 Subprocess 모드** (Resolve DLL 충돌 회피, 안정성 우선)
7. **SAM 모델은 WSL Subprocess 모드** (Windows DLL 충돌 회피, HF 공식 환경)

## 최신 기능 (Phase 3-8)

### Phase 3: ComfyUI 통합 (AI 생성 B-roll)
- **ComfyUIGenerator** (`broll/comfyui_generator.py`): 워크플로우 실행, 출력 수집
- **PromptGenerator** (`ai/prompt_generator.py`): LLM 기반 프롬프트 생성
- **GeneratorBridge** (`broll/generator_bridge.py`): 스톡/AI 생성 통합 레이어
- **설정 UI**: B-roll Source Strategy (Library First, Stock First, AI First)

### Phase 4: OpenCV 비디오 분석
- **VideoAnalyzer** (`adapters/video_analyzer.py`): 비디오 정보 추출, 프레임 샘플링
- **SceneDetector** (`adapters/scene_detector.py`): 히스토그램 기반 장면 전환 감지
  - Bhattacharyya distance 알고리즘 (임계값 10-50, 기본 30)
  - `indexer.smart_frame_sampling()`에서 시간 기반 샘플링 대체
- **VideoQualityChecker** (`broll/quality_checker.py`): 블러/밝기/노이즈 측정
  - 블러: Laplacian 분산 (>100 = 양호)
  - 밝기: 평균 그레이스케일 (50-200 = 양호)
  - 노이즈: Sobel 엣지 강도 표준편차 (<30 = 양호)
  - `direct_generator._generate_stock()`에서 품질 게이트 + 재시도 로직 적용
- **설정 UI**: E. Video Analysis 섹션
  - Scene Detection 토글 + 감도 슬라이더
  - Quality Check 토글 + 최소 품질 슬라이더

### Phase 5: Library Manager UI
- **LibraryManager** (`ui/library_manager.py`): 독립 실행 가능한 DB 관리 도구
  - 클립 목록 조회 (페이지네이션 1000개/페이지, 정렬, FTS5 검색)
  - 썸네일 프리뷰 (OpenCV) + 비디오 재생 (QtMultimedia)
  - 메타데이터 편집 (Folder/Vision Tags, Description) + 명시적 Save
  - 일괄 작업: 다중 선택, 태그 추가, CLIP 재인덱싱 (진행률 표시)
  - 삭제 모드: DB-only 또는 파일 포함 삭제
  - 통계 패널: 총 클립 수, 용량, 평균 길이, 해상도 분포
  - 누락 파일 감지 (빨간색 표시) + 일괄 정리
- **메인 패널 통합**: F. Library 섹션에서 "Open Library Manager" 버튼
- **실행**: `python -m VideoForge.ui.library_manager` (독립 실행) 또는 Resolve 내에서 실행

### Phase 6: SAM3 + SAM Audio (독립 분석 도구)
- **SAM3 Tool** (Misc 탭): WSL SAM3로 비디오 수동 세그멘테이션
  - 프롬프트 기반 객체 감지 ("person", "car", "building" 등)
  - JSON 결과 출력 (frame_idx, timestamp, coverage)
  - 모델 크기 선택 (small/base/large)
- **SAM Audio Tool** (Misc 탭): WSL SAM Audio로 음성 분리
  - 배경음 제거, 특정 음향 추출
  - 모델 크기 선택 (base/large)
- **Subprocess bridge**: sam3_subprocess.py + sam_audio_subprocess.py
- **WSL workers**: sam3_worker.py + sam_audio_worker.py
- **Note**: 자동 파이프라인 통합 없음 (수동 도구로만 사용)








### Phase 7: Library Expansion (Global/Local split)
- **Global/Local Library DB**: shared global and project-local databases loaded together.
- **Search Scope**: choose global/local/both at match time.
- **Settings UI**: Global/Local paths + Search Scope combo box.
- **Library Indexing**: Scan Global/Local with library_type routing.

### Phase 8: Audio-based Sync
- **Use Cases**:
  - Multi-camera sync: same pattern (silence ↔ silence)
  - A-roll/B-roll sync: inverse pattern (voice ↔ silence)
- **SilenceDetector** (`adapters/audio_sync.py`):
  - RMS/dB 기반 무음 구간 감지 (ffmpeg 16kHz mono PCM 추출)
  - 임계값 -40dB, 최소 지속시간 0.3s
  - 반환: `[(start_sec, end_sec), ...]`
- **SyncMatcher** (`core/sync_matcher.py`):
  - Cross-correlation 패턴 매칭 (SciPy/NumPy fallback)
  - Same/Inverse 모드 지원 (bitwise NOT)
  - Confidence 스코어: `max_corr / min(len(ref), len(tgt))`
  - Offset 부호 규칙: 음수 = target이 늦음 (우측 배치)
- **Sync UI** (`ui/sections/sync_section.py`):
  - Reference video picker + Target video list
  - Sync Mode 선택: Same Pattern / Inverse Pattern
  - Auto Sync & Place in Timeline 버튼
  - 신뢰도 경고 표시 (confidence < 0.5)
- **Timeline Placement** (`integrations/resolve_api.py`):
  - Reference를 Track 1에 배치
  - Target을 Track 2+ 에 offset 보정하여 배치
  - `offset_frames = int(-offset_seconds * fps)`
- **Limitations**:
  - 오디오 트랙 필수
  - 연속 음성은 낮은 신뢰도 (무음 패턴 필요)
  - 단일 고정 offset 가정 (drift 미지원)

### Phase 9: Hybrid Sync (Voice + Content)
- **VoiceDetector** (`adapters/voice_detector.py`): Silero VAD speech-only detection (PyTorch optional).
- **ContentMatcher** (`core/content_matcher.py`): librosa-based chroma/onset matching (librosa optional).
- **SyncMatcher modes**: `voice` (speech), `content` (music/melody).

### Phase 11: Auto Multicam Cutter
- **BoundaryDetector** (`multicam/boundary_detector.py`): sentence-based cut boundaries (sentence/fixed/hybrid).
- **AngleScorer** (`multicam/angle_scorer.py`): OpenCV metrics (sharpness, motion, stability, face score).
- **AngleSelector** (`multicam/angle_selector.py`): rule-based angle selection with min-hold/repeat limits.
- **LLMTagger** (`multicam/llm_tagger.py`): optional segment tags (speaking/action/emphasis/neutral).
- **MulticamPlanGenerator** (`multicam/plan_generator.py`): EditPlan generation (`cut_and_switch_angle`).
- **Multicam UI** (`ui/sections/multicam_section.py`): generate preview + approve flow.

## Config Key (Phase 7 - Library Expansion)
```python
Config.get("global_library_db_path", "")   # global shared library DB
Config.get("local_library_db_path", "")    # local project library DB
Config.get("library_search_scope", "both") # "global" | "local" | "both"
```

## Config Key (Phase 8 - Audio Sync)
```python
Config.get("audio_sync_threshold_db", -40.0)  # silence threshold (dB)
Config.get("audio_sync_min_silence", 0.3)     # min silence duration (sec)
Config.get("audio_sync_max_offset", 30.0)     # max search window (sec)
Config.get("audio_sync_resolution", 0.1)      # pattern resolution (sec)
Config.get("audio_sync_min_confidence", 0.5)  # minimum confidence
```

## Config Key (Phase 9 - Voice/Content Sync)
```python
Config.get("audio_sync_voice_threshold", 0.5)      # Silero VAD threshold
Config.get("audio_sync_voice_min_speech_ms", 250)  # min speech duration (ms)
Config.get("audio_sync_voice_min_silence_ms", 100) # min silence between speech (ms)
Config.get("audio_sync_voice_pad_ms", 30)          # padding (ms)
Config.get("audio_sync_content_feature", "chroma") # "chroma" | "onset"
Config.get("audio_sync_content_sample_rate", 22050)
Config.get("audio_sync_content_hop_length", 512)
Config.get("audio_sync_content_n_fft", 2048)
```

## Config Key (Phase 11 - Multicam)
```python
Config.get("multicam_max_segment_sec", 10.0)    # max segment length
Config.get("multicam_min_hold_sec", 2.0)        # minimum hold time
Config.get("multicam_max_repeat", 3)            # max same-angle repeats
Config.get("multicam_closeup_weight", 0.3)      # face weight boost
Config.get("multicam_wide_weight", 0.2)         # motion weight boost
Config.get("multicam_face_detector", "opencv_dnn")  # face detector
Config.get("multicam_face_model_dir", "")       # optional model dir override
Config.get("multicam_boundary_mode", "hybrid")  # sentence/fixed/hybrid
```

## Version History
v2.0.0 (2026-01-02) - UI Redesign: VSCode-style navigation + DaVinci palette
v1.12.0 (2025-01-XX) - Phase 11 Auto Multicam Cutter
