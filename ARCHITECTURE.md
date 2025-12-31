# VideoForge Agent Architecture

## Overview

VideoForge는 DaVinci Resolve용 AI 자동 편집 플러그인으로, 음성 인식, B-roll 매칭, 컷 편집 기능을 통합한 에이전트 기반 시스템입니다.

## Core Components

### 1. Agent System (`VideoForge/agent/`)

#### AgentExecutor (`executor.py`)
**Role**: LLM 기반 에이전트 실행 엔진

- **모델**: Gemini 2.0 Flash (기본) - REST API 호출
- **실행 모드**:
  - `recommend_only`: 읽기 전용, EditPlan만 생성
  - `approve_required`: 수정 작업에 승인 필요 (기본)
  - `full_access`: 자동 실행 (최대 5개 액션)

- **주요 메서드**:
  - `run(state, user_input)`: 메인 실행 루프
  - `_build_prompt(state)`: 시스템 프롬프트 빌드
  - `_parse_agent_response(text)`: JSON 응답 파싱
  - `_call_gemini(prompt)`: Gemini API 호출

- **하드 제약사항**:
  - Resolve API는 Main Thread에서만 호출
  - Whisper 모델 절대 삭제 금지
  - 기본적으로 비파괴 Draft Mode

#### Planner (`planner.py`)
**Role**: 편집 계획 (EditPlan) 스키마 정의 및 파싱

- **ActionType**:
  - `insert_clip`: 클립 삽입
  - `delete_clip`: 클립 삭제
  - `move_clip`: 클립 이동
  - `cut_clip`: 클립 분할
  - `sync_clips`: 오디오 동기화
  - `match_broll`: B-roll 매칭
  - `apply_transition`: 트랜지션 적용
  - `cut_and_switch_angle`: 멀티캠 앵글 전환

- **RiskLevel**: `safe`, `moderate`, `destructive`

- **EditPlan 스키마**:
```json
{
  "request": "사용자 요청",
  "summary": "1-2 문장 요약",
  "actions": [
    {
      "action_type": "action_name",
      "target": "V1|A1|timecode|clip_name",
      "params": {},
      "reason": "사용자 설명"
    }
  ],
  "estimated_clips": 0,
  "risk_level": "safe|moderate|destructive",
  "warnings": []
}
```

#### Tools (`tools.py`)
**Role**: Resolve API 기반 도구 레지스트리

- **데코레이터**: `@videoforge_tool(name, description, parameters)`

- **등록된 도구**:
  - `get_timeline_info()`: 현재 타임라인 메타데이터 반환
  - `insert_clip(path, track_type, track_index, start_frame/seconds/timecode)`: 클립 삽입
  - `delete_clip_at_playhead()`: 플레이헤드 위치 클립 삭제
  - `get_clips_at_playhead(track_types)`: 플레이헤드 위치 클립 반환
  - `sync_clips_from_timeline(reference_path, target_paths, mode)`: 오디오 동기화
  - `match_broll_for_segment(query, duration, limit)`: B-roll 매칭

- **스레드 가드**: `_require_main_thread()`로 Resolve API 호출 시 Main Thread 확인

#### Runtime (`runtime.py`)
**Role**: 세션 상태 관리

- **AgentState**:
  - `session_id`: 세션 식별자
  - `mode`: 실행 모드
  - `conversation`: 대화 기록
  - `pending_plan`: 승인 대기 중인 계획
  - `pending_tool_calls`: 실행 대기 중인 도구 호출
  - `timeline_context`: 타임라인 컨텍스트

#### Memory (`memory.py`)
**Role**: 장기 메모리 저장소 (Phase 10 Week 2+)

- **MemoryRecord**: key-value 쌍
- **MemoryStore**: 인메모리 저장소 (스캐폴딩용)

---

### 2. Core System (`VideoForge/core/`)

#### SegmentStore (`segment_store.py`)
**Role**: 분석 결과 저장소 (SQLite)

- **테이블**:
  - `segments`: 오디오 세그먼트 (t0, t1, type, metadata)
  - `sentences`: 문장 단위 텍스트 (t0, t1, text, confidence, metadata)
  - `matches`: B-roll 매칭 결과 (sentence_id, clip_id, score, metadata)
  - `draft_order`: Draft Mode 순서 (uid, position)
  - `artifacts`: 메타데이터 저장 (key, value)

- **주요 메서드**:
  - `save_segments/` `sentences/` `matches()`: 데이터 저장
  - `update_sentences()`: 문장 업데이트
  - `replace_sentence()` / `replace_all_sentences()`: 문장 교체
  - `clear_analysis_data()`: 분석 데이터 초기화
  - `save_draft_order()` / `get_draft_order()`: Draft Mode 순서 관리

- **Schema Version**: 1.1 (버전 2)

#### TimelineBuilder (`timeline_builder.py`)
**Role**: 타임라인 데이터 구조 빌더

- **데이터 구조**:
```json
{
  "schema_version": "1.1",
  "pipeline_version": "0.1.0",
  "metadata": {...},
  "settings": {...},
  "tracks": [
    {"id": "V1", "type": "video_main", "clips": [...]},
    {"id": "V4", "type": "video_broll", "clips": [...]}
  ],
  "pipeline_info": {...}
}
```

- **주요 메서드**:
  - `build(main_video_path)`: 전체 타임라인 빌드
  - `_build_main_track()`: 메인 비디오 트랙 (A-roll)
  - `_build_segment_map()`: 세그먼트 맵 생성 (overlap, jitter 지원)

- **SegmentMap**: `segment_id`, `t0`, `t1`, `timeline_start`

#### SyncMatcher (`sync_matcher.py`)
**Role**: 오디오 패턴 기반 동기화 (Phase 8)

- **지원 모드**:
  - `same`: 동일 패턴 (멀티캠 동기화)
  - `inverse`: 반대 패턴 (A-roll/B-roll 동기화)
  - `voice`: Silero VAD 음성 감지 (Phase 9)
  - `content`: librosa chroma/onset 매칭 (Phase 9)

- **알고리즘**: Cross-correlation (SciPy/NumPy fallback)

- **신뢰도**: `max_corr / min(len(ref), len(tgt))`

- **Offset 부호**: 음수 = target이 늦음 (우측 배치)

#### ContentMatcher (`content_matcher.py`)
**Role**: 음악/멜로디 기반 동기화 (Phase 9)

- **특징 추출**:
  - `chroma`: 화음 진행
  - `onset`: 비트/리듬

---

### 3. B-roll System (`VideoForge/broll/`)

#### BrollMatcher (`matcher.py`)
**Role**: 문장 기반 B-roll 매칭

- **검색 전략**:
  1. QueryGenerator: 문장 → primary/fallback/boost 쿼리 생성
  2. LibraryDB.hybrid_search(): FTS5 + Vector 검색
  3. CooldownManager: 최근 클립 재사용 방지 (쿨다운 + 페널티)
  4. VisualSimilarityFilter: 시각적 유사성 필터링 (CLIP 임베딩 코사인 유사도)
  5. ScriptIndexSelector: script_index 태그 우선 선택

- **QueryGenerator**:
  - 불용어 제거 (`the`, `a`, `and` 등)
  - 토큰 스코어링 (길이, 카멜케이스, 영문/숫자 혼합)
  - 키워드 번역 (영어 ↔ 한국어)
  - `top_n` 토큰 선택 (기본 6개)

- **CooldownManager**:
  - `cooldown_sec`: 재사용 금지 기간
  - `penalty`: 쿨다운 클립 스코어 페널티 (기본 0.4)

- **VisualSimilarityFilter**:
  - CLIP 임베딩 코사인 유사도
  - `threshold`: 0.85 (기본)
  - 최근 N개 클립의 임베딩 저장

#### Placer (`placer.py`)
**Role**: B-roll 클립 타임라인 배치

- **설정**:
  - `max_overlay_duration`: 최대 오버레이 시간
  - `fit_mode`: `cover` / `contain` / `fill`
  - `crossfade`: 크로스페이드 시간

- **매핑 함수**:
  - `_map_time_to_timeline_fast()`: bisect 기반 빠른 시간 매핑

#### LibraryDB (`db.py`)
**Role**: Global/Local 라이브러리 DB (Phase 7)

- **테이블**:
  - `clips`: 클립 메타데이터
  - `tags`: FTS5 태그 인덱스
  - `embeddings`: 벡터 임베딩

- **메서드**:
  - `hybrid_search()`: FTS5 + Vector 하이브리드 검색
  - `get_clips_by_ids()`: ID로 클립 조회
  - `get_embeddings_by_ids()`: ID로 임베딩 조회
  - `set_cooldown()`: 쿨다운 페널티 설정

#### Indexer (`indexer.py`)
**Role**: 라이브러리 인덱싱 + 장면 기반 샘플링 (Phase 4-B)

- **장면 감지 우선**: 시간 기반 샘플링보다 25% 정확도 향상
- **Smart frame sampling**: SceneDetector 기반 샘플링

#### Matcher (`matcher.py`)
**Role**: 문장-클립 매칭 로직

#### Orchestrator (`orchestrator.py`)
**Role**: B-roll 생성/매칭 오케스트레이션

#### QualityChecker (`quality_checker.py`)
**Role**: OpenCV 품질 검증 (Phase 4-C)

- **품질 지표**:
  - 블러: Laplacian 분산 (>100 = 양호)
  - 밝기: 평균 그레이스케일 (50-200 = 양호)
  - 노이즈: Sobel 엣지 강도 표준편차 (<30 = 양호)

- **재시도 로직**: 최대 3회 재시도

#### DirectGenerator (`direct_generator.py`)
**Role**: 다이렉트 B-roll 생성 + 품질 검증 (Phase 4-C)

- **품질 게이트 + 재시도**: 저품질 스톡/AI 자동 필터링

#### ComfyUIGenerator (`comfyui_generator.py`)
**Role**: ComfyUI 워크플로우 실행 (Phase 3-A/B)

---

### 4. Adapters (`VideoForge/adapters/`)

#### VideoAnalyzer (`video_analyzer.py`)
**Role**: OpenCV 비디오 분석 (Phase 4-A)

- **메서드**:
  - `get_video_info(video_path)`: width, height, fps, frame_count, duration
  - `extract_frame_at_time(video_path, timestamp)`: 프레임 추출

#### SceneDetector (`scene_detector.py`)
**Role**: 히스토그램 기반 장면 전환 감지 (Phase 4-B)

- **알고리즘**: Bhattacharyya distance
- **임계값**: 10-50 (기본 30)
- **Subprocess 모드**: `opencv_subprocess.py`로 실행 (Resolve DLL 충돌 회피)

#### AudioAdapter (`audio_adapter.py`)
**Role**: Whisper 기반 오디오 전사

- **모델**: faster-whisper (large-v3, GPU)
- **출력**: SRT, JSON

#### AudioSync (`audio_sync.py`)
**Role**: 무음 구간 감지 (Phase 8-A)

- **방식**: RMS/dB 기반 (ffmpeg 16kHz mono PCM 추출)
- **임계값**: -40dB (기본)
- **최소 지속시간**: 0.3s

#### VoiceDetector (`voice_detector.py`)
**Role**: Silero VAD 음성 감지 (Phase 9)

- **모델**: Silero VAD (PyTorch optional)
- **출력**: speech-only 타임스탬프

#### SAM3Worker / SAMAudioWorker
**Role**: SAM3 + SAM Audio WSL subprocess bridge (Phase 6)

- **목적**: Windows DLL 충돌 회피
- **모델**: HF 공식 환경

---

### 5. Integrations (`VideoForge/integrations/`)

#### ResolveAPI (`resolve_api.py`)
**Role**: DaVinci Resolve Scripting API 래퍼

- **주요 메서드**:
  - `get_current_project()`: 현재 프로젝트
  - `get_current_timeline()`: 활성 타임라인
  - `get_selected_clips()` / `get_selected_items()`: 선택된 클립
  - `get_items_at_playhead()`: 플레이헤드 위치 클립
  - `get_clip_at_playhead()`: 플레이헤드 클립
  - `insert_clip_at_position()`: 클립 삽입
  - `replace_timeline_item()`: 클립 교체
  - `export_clip_audio()`: 오디오 추출
  - `import_subtitles_into_timeline()`: 자막 가져오기

- **스레드 가드**: `_ensure_main_thread()`로 Main Thread 확인
  - 경고 또는 에러 (`VIDEOFORGE_STRICT_RESOLVE_THREAD` 환경변수)

- **멀티캠 컷**:
  - `apply_multicam_cuts(plan)`: cut_and_switch_angle 액션 실행
  - `_build_multicam_timeline()`: 새 타임라인 생성 (소스 + 프로그램 트랙)
  - `_build_multicam_program_timeline()`: 프로그램 전용 타임라인
  - `_export_multicam_edl()`: EDL 내보내기 (폴백)

- **자막 관리**:
  - `create_subtitles_from_audio()`: Resolve Studio 자동 자막 생성
  - `get_subtitle_items()`: 자막 아이템 조회
  - `export_subtitles_as_sentences()`: 자막 → 문장 변환

- **시간 변환**:
  - `_timecode_to_frames()`: 타임코드 → 프레임
  - `_frames_to_timecode()`: 프레임 → 타임코드
  - `_get_item_range()`: 아이템 범위 (프레임)
  - `get_item_range_seconds()`: 아이템 범위 (초)

---

### 6. UI Sections (`VideoForge/ui/sections/`)

#### AgentSection (`agent_section.py`)
**Role**: Agent 채팅 인터페이스

- **기능**:
  - 채팅 입력/출력
  - EditPlan 승인/거절
  - Tool 실행 모드 전환
  - Timeline 컨텍스트 로드

#### LibrarySection (`library_section.py`)
**Role**: Library DB 경로 관리

- **Global/Local DB 경로 설정**
- **검색 범위 선택**: `global`, `local`, `both`

#### MatchSection (`match_section.py`)
**Role**: B-roll 매칭 실행 UI

#### SyncSection (`sync_section.py`)
**Role**: Audio Sync UI (Phase 8-C)

- **기능**:
  - Reference video picker
  - Target video list
  - Sync Mode: Same Pattern / Inverse Pattern
  - Auto Sync & Place 버튼
  - 신뢰도 경고 표시

#### MulticamSection (`multicam_section.py`)
**Role**: Auto Multicam Cutter UI (Phase 11)

- **기능**:
  - Generate preview + approve flow
  - Boundary mode: sentence/fixed/hybrid
  - Min-hold, max-repeat 설정

#### SettingsSection (`settings_section.py`)
**Role**: E. Video Analysis, F. Library (Phase 4-5)

- **Video Analysis**:
  - Scene Detection 토글 + 감도 슬라이더
  - Quality Check 토글 + 최소 품질 슬라이더

#### MiscSection (`misc_section.py`)
**Role**: SAM Tools, WSL 환경, Misc 도구 (Phase 6)

---

### 7. Multicam (`VideoForge/multicam/`)

#### BoundaryDetector (`boundary_detector.py`)
**Role**: 문장 기반 컷 바운더리 (Phase 11)

- **모드**: `sentence`, `fixed`, `hybrid`

#### AngleScorer (`angle_scorer.py`)
**Role**: OpenCV 기반 앵글 스코어링 (Phase 11)

- **지표**: sharpness, motion, stability, face score

#### AngleSelector (`angle_selector.py`)
**Role**: 룰 기반 앵글 선택 (Phase 11)

- **제한**: min-hold, max-repeat

#### LLMTagger (`llm_tagger.py`)
**Role**: 세그먼트 태깅 (Phase 11)

- **태그**: speaking, action, emphasis, neutral

#### PlanGenerator (`plan_generator.py`)
**Role**: EditPlan 생성 (Phase 11)

- **액션**: `cut_and_switch_angle`

---

## Config Keys

### Agent
```python
Config.get("agent_api_key", "")           # Agent API key
Config.get("agent_model", "gemini-2.0-flash-exp")  # Agent model
Config.get("agent_max_tokens", 4096)      # Max tokens
Config.get("agent_max_auto_actions", 5)    # Max auto actions
```

### Library Expansion (Phase 7)
```python
Config.get("global_library_db_path", "")   # Global shared library DB
Config.get("local_library_db_path", "")    # Local project library DB
Config.get("library_search_scope", "both") # "global" | "local" | "both"
```

### Audio Sync (Phase 8)
```python
Config.get("audio_sync_threshold_db", -40.0)  # Silence threshold (dB)
Config.get("audio_sync_min_silence", 0.3)     # Min silence duration (sec)
Config.get("audio_sync_max_offset", 30.0)     # Max search window (sec)
Config.get("audio_sync_resolution", 0.1)      # Pattern resolution (sec)
Config.get("audio_sync_min_confidence", 0.5)  # Minimum confidence
```

### Voice/Content Sync (Phase 9)
```python
Config.get("audio_sync_voice_threshold", 0.5)      # Silero VAD threshold
Config.get("audio_sync_voice_min_speech_ms", 250)  # Min speech duration (ms)
Config.get("audio_sync_voice_min_silence_ms", 100) # Min silence between speech (ms)
Config.get("audio_sync_voice_pad_ms", 30)          # Padding (ms)
Config.get("audio_sync_content_feature", "chroma")  # "chroma" | "onset"
Config.get("audio_sync_content_sample_rate", 22050)
Config.get("audio_sync_content_hop_length", 512)
Config.get("audio_sync_content_n_fft", 2048)
```

### Multicam (Phase 11)
```python
Config.get("multicam_max_segment_sec", 10.0)    # Max segment length
Config.get("multicam_min_hold_sec", 2.0)        # Minimum hold time
Config.get("multicam_max_repeat", 3)            # Max same-angle repeats
Config.get("multicam_closeup_weight", 0.3)      # Face weight boost
Config.get("multicam_wide_weight", 0.2)         # Motion weight boost
Config.get("multicam_face_detector", "opencv_dnn")  # Face detector
Config.get("multicam_face_model_dir", "")       # Optional model dir override
Config.get("multicam_boundary_mode", "hybrid")  # sentence/fixed/hybrid
Config.get("multicam_audio_mode", "per_cut")     # "per_cut" | "fixed_track"
Config.get("multicam_audio_track", 1)           # Fixed audio track index
Config.get("multicam_edl_output_dir", "")       # EDL output directory
```

---

## Key Design Principles

1. **Main Thread Only**: Resolve API는 Main Thread에서만 호출 (Worker 사용 금지)
2. **Model Persistence**: Whisper 모델 절대 삭제 금지 (del/gc.collect → 크래시)
3. **Draft Mode**: 비파괴 모드 기본 (UID 추적, 렌더 시점 반영)
4. **OpenCV First**: 장면 감지 우선 (시간 기반 샘플링보다 25% 정확도 향상)
5. **Quality + Retry**: 품질 검증 + 재시도 (최대 3회)
6. **Subprocess Mode**: OpenCV, SAM은 Subprocess 모드 (DLL 충돌 회피, 안정성 우선)

---

## Execution Flow (Example: B-roll Matching)

1. **사용자 입력**: "이 문장에 맞는 B-roll 찾아줘"
2. **AgentExecutor.run()**: Gemini API 호출
3. **Planner**: EditPlan 생성 (`action_type: "match_broll"`)
4. **Tools.match_broll_for_segment()**: BrollMatcher 호출
5. **BrollMatcher.match()**:
   - QueryGenerator.generate(): 쿼리 생성
   - LibraryDB.hybrid_search(): FTS5 + Vector 검색
   - CooldownManager.apply_penalty(): 쿨다운 적용
   - VisualSimilarityFilter.is_too_similar(): 시각적 유사성 필터링
6. **Placer.build_broll_clips()**: 타임라인 배치
7. **ResolveAPI**: 타임라인에 클립 삽입

---

## Error Handling

- **Timeline not found**: 타임라인 로드 요청
- **Main thread required**: UI 흐름 사용 지시
- **Clip not found**: 정확한 클립 이름/트랙/타임코드 요청
- **Tool execution blocked**: EditPlan 생성 요청

---

## Testing

- **Unit Tests**: `tests/` 디렉토리
- **Integration Tests**: Resolve API 호출 테스트
- **Lint/Typecheck**: `npm run lint`, `npm run typecheck` (또는 해당 명령어)

---

## Version

**v1.12.0** (2025-01-XX) - Phase 11 Auto Multicam Cutter
