# VideoForge: AI 기반 영상 편집 자동화 파이프라인 제안서 (for CODEX)

## 1. 프로젝트 개요 및 현재 상태 분석

VideoForge는 "Local-first video editing automation"을 목표로 하는 MVP(Minimum Viable Product) 단계의 프로젝트입니다. 현재 구조는 매우 정교하게 모듈화되어 있으며, 실제 편집 워크플로우(분석 -> 매칭 -> 타임라인 생성 -> 내보내기)를 충실히 따르고 있습니다.

### 핵심 강점 (Strengths)
- **모듈형 아키텍처**: `adapters`, `core`, `integrations` 등으로 명확히 분리되어 확장성이 뛰어남.
- **워크플로우 정립**: DaVinci Resolve와의 직접 연동 및 `timeline.json`을 통한 표준 인터페이스 확보.
- **데이터 관리**: SQLite 기반의 `SegmentStore`와 `LibraryDB`를 통해 프로젝트별/라이브러리별 데이터 관리 체계가 잡혀 있음.
- **기본 분석 능력**: 오디오 무음 구간 기반의 세그먼트 분리 및 Whisper를 활용한 텍스트 변환 기능 작동.

### 현재 한계 및 개선 필요점 (Gaps)
- **시각 분석 부재**: 영상 임베딩(`encode_video_clip`)이 현재 랜덤 벡터를 반환하는 플레이스홀더 상태임.
- **미사용 모듈**: `smart_frame_sampling`과 같은 유용한 로직이 구현되어 있으나 실제 인덱싱 과정에는 통합되지 않음.
- **매칭 정밀도**: 텍스트 검색(FTS)에 의존도가 높으며, 영상의 실제 시각적 내용과 문서 사이의 매칭은 아직 초기 단계임.

---

## 2. 향후 발전 로드맵 제안

### Phase 1: 시각적 내용 인식 강화 (최우선 과제)
- **실제 CLIP/VLM 통합**: `encode_video_clip`을 실제 CLIP 모델로 교체하고, `smart_frame_sampling`을 활용해 영상의 주요 프레임을 분석.
- **자동 Vision Tagging**: B-roll 스캔 시 이미지 캡셔닝 모델을 사용하여 `description` 및 `vision_tags`를 자동으로 생성, 하이브리드 검색의 성능을 극대화.
- **OCR 및 객체 탐지**: 영상 내 텍스트나 특정 객체를 인식하여 검색 속성으로 추가.

### Phase 2: 지능형 편집 로직 도입
- **맥락 기반 매칭 (Contextual Matching)**: 단순 단어 일치가 아닌, 전체 문장의 의미와 영상의 분위기(Sentiment)를 매칭하는 LLM 기반 매칭 엔진 도입.
- **리듬 기반 컷 편집**: 오디오 비트(Beat)나 화자의 말하기 속도에 맞춰 B-roll의 진입/진출 지점을 최적화하는 기능.
- **중복 방지 및 다양성 확보**: 동일 프로젝트 내에서 유사한 B-roll이 반복되지 않도록 하는 '다양성 필터' 강화.

### Phase 3: 사용자 경험(UX) 및 인터페이스 확장
- **Standalone Web UI**: Resolve 내부뿐만 아니라 별도의 웹 대시보드(Gradio 등)를 통해 편집 결과를 미리 보고 수정할 수 있는 환경 제공.
- **실시간 피드백 루프**: 사용자가 제안된 B-roll을 거절하거나 확정하면 이를 학습 데이터로 활용하여 매칭 성능을 점차 개선.
- **클라우드/API 연동**: 로컬 라이브러리뿐만 아니라 Pexels, Unsplash 등 외부 스토리지 API와의 연동 지원.

---

## 3. CODEX를 위한 실행 과제 (Action Items)

1. **`VideoForge/adapters/embedding_adapter.py` 업데이트**: 
   - `open-clip`을 활용한 실제 영상 특징 추출 로직 활성화.
2. **`VideoForge/broll/indexer.py` 리팩토링**: 
   - `smart_frame_sampling`을 인덱싱 루프에 결합하여 다중 프레임 기반 임베딩 구현.
3. **`VideoForge/core/transcriber.py` 강화**: 
   - 화자 분리(Diarization) 및 감정 분석 태그 추가.
4. **테스트 데이터셋 확보**: 
   - 실제 편집 상황에서의 매칭 정확도를 측정할 수 있는 벤치마킹 환경 구축.

---

**결론**: VideoForge는 이미 훌륭한 골격을 갖추고 있습니다. 이제 "가짜 데이터"를 "AI 기반의 실제 데이터"로 교체하고, 편집자의 의도를 더 깊이 이해하는 지능형 레이어를 추가한다면 실질적인 생산성 도구로 자리잡을 것입니다.
