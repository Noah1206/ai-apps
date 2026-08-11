아래 블록은 **Audio Lab v0.1이 완성된 직후부터 최종 Busan Dialect Tutor Engine까지 개발하도록 Codex에 주는 단일 실행 스크립트**야. 네 문서에 정의된 전체 계층형 아키텍처와 Gate 0~15를 유지한다. Gate는 장기 기술 의존성과 연구 성숙도를 나타내며, 제품 통합은 모든 연구 Gate가 끝날 때까지 기다리지 않고 `Offline Vertical Slice v0` 트랙에서 조기에 시작한다.

Codex의 Master Identity 프롬프트 다음에 아래 내용을 그대로 붙여 넣으면 된다.

````md
# POST-AUDIO-LAB EXECUTION DIRECTIVE
## Busan Dialect Tutor Engine

너는 현재 `CODEX PRIME DIRECTIVE` 아래에서 동작한다.

이제 비전 설명이나 일반적인 조언을 반복하지 마라.
현재 저장소를 직접 조사하고, 코드를 수정하고, 테스트하고,
실제 음성과 고정 Benchmark로 결과를 측정하면서
Busan Dialect Tutor Engine을 단계적으로 완성하라.

---

# 0. 현재 프로젝트 상태

현재 다음 산출물은 완성된 것으로 가정한다.

`Busan Speech Research Lab v0.1`

Audio Lab에는 최소한 다음 기능이 존재해야 한다.

- 오디오 업로드 또는 마이크 녹음
- 원본 오디오 보존
- 16kHz mono 파생본 생성
- 오디오 길이, Sample Rate, 채널, RMS, Peak 검사
- Clipping Ratio와 Silence Ratio 검사
- 파형 표시
- Mel Spectrogram 표시
- F0 표시
- Surface Text 입력
- Normalized Meaning 별도 입력
- 화자와 녹음 환경 메타데이터
- 공통 Utterance Schema
- 모델 Adapter Interface
- 모델별 결과 표시
- 오류 태그 저장
- 실험 ID, 모델 버전, 데이터 버전 저장

이 항목 중 실제로 존재하지 않는 기능은
구현 완료로 가정하지 말고 저장소 검사 결과에 명시하라.

Audio Lab을 처음부터 다시 만들지 마라.

기존 코드를 보존하면서 누락 기능만 보완하고,
그 위에 모델과 평가 시스템을 추가하라.

---

# 1. 최종 미션

Audio Lab 이후 다음 전체 시스템을 완성한다.

```text
사용자 음성
    ↓
Real-time Media Gateway
WebRTC · VAD · AEC · Noise Suppression
Resampling · Buffer · Barge-in
    ↓
Speech Understanding Platform
├─ Streaming ASR
├─ Surface-form ASR
├─ Direct IPA Phonetic Model
├─ IPA / Acoustic Forced Alignment
├─ Pronunciation Assessment
├─ Prosody Analysis
└─ Dialect Acoustic Model
    ↓
Dialect and Linguistic Understanding
├─ Surface Form
├─ Normalized Meaning
├─ Recommended Busan Form
├─ Intent · Slots · Grammar
├─ Relationship · Register · Formality
└─ Dialect Naturalness
    ↓
Tutor Coordinator
├─ Scenario Engine
├─ Learner Model
├─ Teaching Policy
├─ Curriculum Engine
├─ Dialect Knowledge Base
└─ Confidence Gate
    ↓
Dialogue Planner
    ↓
Structured Speech Plan
    ↓
Busan Speech Generator
├─ Text Normalization
├─ Dialect-aware G2P
├─ IPA
├─ Prosody Planner
├─ Acoustic / Speech-token Model
├─ Vocoder / Codec Decoder
└─ Streaming TTS
    ↓
실제 부산 사투리 음성
    ↓
Learner Model 업데이트
    ↓
실패 데이터 검수 및 재학습 후보 저장
````

최종 결과는 단일 거대 체크포인트일 필요가 없다.

여러 전문 모델, 규칙 엔진, 데이터 시스템과  
오케스트레이터가 하나의 부산 AI 튜터처럼 동작하면 된다.

---

# 2. 개발 환경 전제

사용자는 macOS의 VS Code를 주 개발 환경으로 사용한다.

## 로컬 Mac에서 수행

- 코드 작성
    
- Git
    
- Codex 사용
    
- Audio Lab UI
    
- 데이터 정리
    
- 오디오 검사
    
- CPU 기반 평가
    
- 작은 단위 테스트
    
- 문서와 실험 보고서
    
- 원격 GPU 제어
    

## 원격 Linux NVIDIA GPU에서 수행

- 한국어 Surface ASR 추론과 향후 파인튜닝
    
- RNNT/TDT 학습
    
- Direct IPA Model 학습
    
- TTS 파인튜닝
    
- 대규모 Batch Inference
    
- GPU 추론 성능 평가
    

로컬 환경에 CUDA가 없다는 이유로  
프로젝트 전체를 실행 불가능하게 만들지 마라.

CPU로 가능한 기능과 GPU가 필요한 기능을 분리하라.

원격 GPU 작업은 VS Code Remote SSH 또는  
재현 가능한 SSH 명령으로 수행할 수 있게 하라.

기존 프로젝트가 `uv`, Poetry, Conda 또는 다른 도구를 사용한다면  
저장소를 먼저 확인하고 기존 방식을 유지하라.

사용자 승인 없이 패키지 관리 방식을 마이그레이션하지 마라.

---

# 3. 최우선 행동

새 코드를 작성하기 전에 다음을 수행하라.

## 3.1 저장소 조사

다음을 확인한다.

- Git 상태
    
- 현재 Branch
    
- 디렉터리 구조
    
- Python 버전
    
- 패키지 관리 방식
    
- Audio Lab 진입점
    
- 공통 Schema 위치
    
- 오디오 처리 코드
    
- 테스트 위치
    
- Benchmark 존재 여부
    
- Model Adapter Interface
    
- 실험 기록 방식
    
- 데이터 저장 위치
    
- 개인정보 또는 동의 필드
    
- README 실행 명령
    

## 3.2 현재 상태 보고서 생성

다음 파일을 생성하거나 갱신한다.

```text
docs/status/project-state.md
docs/status/post-audio-lab-roadmap.md
docs/architecture/current-system.md
```

`project-state.md`에는 다음을 기록한다.

```text
완성된 기능
부분적으로 구현된 기능
누락된 기능
실행 가능한 명령
실패하는 테스트
데이터 상태
모델 상태
현재 병목
즉시 필요한 외부 자원
```

## 3.3 Audio Lab 동결

기존 테스트가 통과하고 핵심 기능이 동작하면  
현재 상태를 Git Tag 또는 명시적 버전으로 기록한다.

예:

```text
audio-lab-v0.1
```

Tag를 생성하기 전에 Git 상태와 사용자 권한을 확인하라.

원격 Push가 필요하다면 사용자 승인 없이 수행하지 마라.

---

# 4. 공통 개발 원칙

## 4.1 Gate 기반 개발

다음 Gate는 최종 아키텍처의 기술 의존성과 연구 성숙도 기준이다.
Gate 번호와 정의는 유지하지만 제품 기능의 구현 착수 순서를 강제하는 단일 직렬 목록으로 사용하지 않는다.

```text
Gate 0  Audio Lab Audit and Freeze
Gate 1  Pretrained Korean Surface ASR Baseline
Gate 2  Baseline 결과 기반 Surface ASR Improvement
Gate 3  Streaming ASR
Gate 4  Dialect-aware G2P and Target IPA
Gate 5  Direct IPA Phonetic Model
Gate 6  Forced Alignment and Pronunciation Diagnosis
Gate 7  Prosody and Dialect Acoustic Model
Gate 8  Speech Understanding Core
Gate 9  Busan Domain Brain
Gate 10 Tutor Coordinator
Gate 11 Dialogue Planner and Structured Speech Plan
Gate 12 Busan TTS Integration
Gate 13 Offline Vertical Slice
Gate 14 Real-time WebRTC Integration
Gate 15 Data Flywheel, MLOps and Production Operations
```

각 Gate는 코드가 실행되는 것으로 끝나지 않는다.

다음이 모두 있어야 통과한다.

```text
작동 코드
자동 테스트
실행 명령
입출력 예시
고정 Benchmark 결과
실패 사례
성능 지표
알려진 한계
코드·데이터·모델 버전
다음 실험
```

Gate 통과 조건을 만족하면 불필요한 확인 질문 없이  
현재 실패 기준에 따라 다음 우선순위 작업으로 진행할 수 있다.

Gate가 실패하면 실패 원인을 분류하고,  
가장 작은 수정으로 다시 검증하라.

## 4.2 두 개의 실행 트랙

현재부터 제품 흐름 검증과 핵심 모델 연구를 병행한다.

### Track A: 제품 Vertical Slice

기존 모델과 단순 로직으로 다음 파일 기반 경로를 가능한 한 빨리 완성한다.

```text
사용자 WAV
→ 기존 ASR Provider
→ 단순 Surface·방언 표현 평가
→ 구조화된 LLM Tutor 판단
→ 기존 또는 가볍게 적용한 TTS Provider
→ 부산 응답 WAV
```

이 경로는 최종 ASR, IPA, Prosody 또는 TTS 품질을 증명하지 않는다.
전체 학습 경험과 시스템 계약을 검증하는 `Offline Vertical Slice v0`다.

최소 구성:

```text
ASRProvider
SimpleAssessmentService
TutorProvider
TTSProvider
OfflineSpeechPipeline
```

모든 외부 모델, API와 단순 규칙은 안정적인 Provider 또는 서비스 인터페이스 뒤에 둔다.
부산 특화 구현이 준비되면 Pipeline 호출부가 아니라 Provider 내부 구현을 교체한다.

### Track B: 핵심 모델 연구

Audio Lab과 고정 Benchmark로 모델 후보와 개선 실험을 평가한다.

현재 작업 순서:

```text
TASK-002 RIVA Korean Conformer-CTC Baseline                         완료
TASK-003A Surface ASR Evaluation Calibration                       완료
TASK-003B Nemotron 3.5 Pretrained Surface ASR Baseline Integration 완료(제한 있음)
TASK-003C Nemotron Inference Configuration Validation              완료
TASK-004 Busan ASR Training Dataset Foundation                     진행(Phase A 완료)
TASK-005 Nemotron Busan Fine-tuning Pilot                          후보 보류
```

```text
별도 수집 데이터
→ train/validation 계약
→ Benchmark와 speaker/audio lineage 분리
→ 라벨 검수
→ split·중복·누출 검증
→ 학습 파이프라인 smoke test
→ TASK-005 모델 후보 재판정
```

Benchmark 데이터는 평가 전용이며 학습 또는 파인튜닝에 사용하지 않는다.

### 현재 우선순위와 선택 규칙

1. 완료된 TASK-002 RIVA Baseline과 TASK-003A 평가 revision을 보존한다.
2. 완료된 TASK-003B Nemotron Prediction, 사람 검수와 Baseline Report를 보존한다.
3. 완료된 TASK-003C raw output, Adapter trace와 판정 보고서를 보존한다.
4. `TASK-004` Phase A에서 Benchmark와 분리된 부산 ASR 학습 데이터 계약을
   구현했으며, 다음으로 새 학습 동의 발화를 수집·검수한다.
5. 데이터 기반이 준비된 뒤 RIVA, Nemotron 또는 다른 후보 중 TASK-005 대상을 다시 판단한다.
6. 이후 Offline Vertical Slice와 사용자 테스트에서 관찰된 가장 큰 실패로 다음 연구 우선순위를 정한다.

Whisper Cross-model Comparison은 폐기하지 않고 현재 우선순위에서 보류한다.

```text
ASR 방언 보존 실패 → Surface ASR 개선
발음 피드백 부정확 → IPA·Forced Alignment
부산 억양 부족 → TTS·Prosody
교육 판단 부적절 → Teaching Policy·Learner Model
응답 지연 → Streaming
```

ASR, IPA, Forced Alignment, Prosody와 자체 TTS가 모두 완성될 때까지 통합을 미루지 않는다.
동시에 모든 연구 모듈을 구현하지 않는다.

---

# 5. 핵심 모델 연구의 병렬 전문 트랙

이 절은 Track B 안에서 듣기, 말하기, 교육 지능을 나누는 장기 전문 트랙이다.
앞 절의 제품 Track A와 연구 Track B 구분을 대체하지 않는다.

다음 세 트랙을 병렬로 운영한다.

## 연구 전문 영역 1: 듣는 귀

```text
Surface ASR
→ Streaming ASR
→ Direct IPA
→ Forced Alignment
→ Pronunciation Assessment
→ Prosody
→ Dialect Acoustic
→ Speech Understanding Core
```

## 연구 전문 영역 2: 말하는 입

```text
TTS Data Schema
→ Reference Recording Plan
→ Text Normalization
→ Dialect G2P
→ Target IPA
→ TTS Baseline
→ Prosody Conditioning
→ Conversation / Reference Modes
→ Streaming TTS
```

## 연구 전문 영역 3: 교사의 두뇌와 플랫폼

```text
Common Schema
→ Error Taxonomy
→ Dialect Knowledge Base
→ Scenario Engine
→ Teaching Policy
→ Learner Model
→ Dialogue Planner
→ Structured Speech Plan
→ Realtime Orchestrator
→ Client Integration
```

세 트랙은 동일한 핵심 부산 문장과 언어 기준을 사용한다.

```text
동일한 부산 문장
├─ Surface ASR Reference
├─ Normalized Meaning
├─ Target IPA Candidates
├─ Prosody Reference
├─ TTS Reference
└─ Teaching Content
```

트랙 간 Schema를 중복 정의하지 마라.

---

# 6. Gate 0: Audio Lab Audit and Freeze

## 목표

Audio Lab이 이후 모든 음성 모델을 테스트할 수 있는  
안정적인 연구 인터페이스인지 확인한다.

## 필수 검증

- 샘플 오디오 업로드 가능
    
- 마이크 녹음 가능
    
- 원본과 파생본 경로 기록
    
- 처리 실패가 명확하게 표시됨
    
- 공통 Utterance Schema 저장
    
- 모델 Adapter 추가 가능
    
- 결과 JSON Export 가능
    
- 모델 A/B 비교 가능
    
- 실험과 모델 버전 기록 가능
    
- 테스트 재현 가능
    

## 추가할 수 있는 인터페이스

```python
class SpeechModelAdapter(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def model_version(self) -> str: ...

    def predict(self, audio_path: Path) -> SpeechPrediction: ...
```

공통 모델 출력에는 최소한 다음이 있어야 한다.

```json
{
  "utterance_id": "utt_000001",
  "model_name": "example_model",
  "model_version": "v0.1",
  "prediction": "",
  "confidence": null,
  "latency_ms": null,
  "timestamps": [],
  "metadata": {}
}
```

## 통과 조건

Audio Lab에 새로운 Adapter를 추가할 때  
기존 UI와 Schema를 수정하지 않아도 된다.

---

# 7. Gate 1: Pretrained Korean Surface ASR Baseline

## 목표

사전학습 한국어 Surface ASR을 파인튜닝하지 않고 고정 부산 Benchmark에
실행한다. 첫 후보는 NVIDIA Korean Conformer-CTC이며, 특정 아키텍처 이름을
유지하기 위해 한국어 미지원 모델을 사용하지 않는다.

현재 첫 연구 질문:

> 사전학습 한국어 ASR은  
> 부산 원어민과 학습자의 발화를  
> 어떤 유형으로 틀리는가?

## 첫 NVIDIA 후보와 검증 상태

```text
model_provider: NVIDIA
model_name: RIVA Conformer ASR Korean
model_family: Conformer-CTC
model_version: deployable_v1.0
checkpoint_identifier: nvidia/tao/speechtotext_ko_kr_conformer:deployable_v1.0
decoder_type: CTC greedy
fine_tuned: false
language: ko-KR
```

공개 NVIDIA NGC 페이지에서 모델명, 언어, Conformer-CTC 구조, 버전,
약 120M 파라미터, Riva Quick Start 사용 경로와 Riva 라이선스를 확인했다.
이 항목은 일반 Python용 NeMo 체크포인트가 아니라 Riva 배포 모델로 안내된다.
NGC 파일 브라우저와 다운로드에는 로그인이 필요하므로 정확한 배포 파일명,
파일 hash, 호환 Riva 서버 버전은 `pending verification`이다. 확인 전에는
`.nemo` 파일명이나 NeMo `restore_from` 경로를 코드에 넣지 않는다.

## Surface ASR의 역할

- 실제 사용한 표현 보존
    
- 부산 표현의 표준어 과보정 탐지
    
- 발화 종료 후 정밀 전사
    
- Dialect G2P와 IPA 분석의 문장 기준 제공
    
- 오류 분석과 데이터 정제
    

다음 필드를 절대로 합치지 마라.

```json
{
  "surface_text": "국밥 하나 주이소",
  "normalized_meaning": "국밥 하나 주세요"
}
```

## 입력

```text
16kHz mono WAV
Reference Surface Text
Normalized Meaning
Speaker ID
Speaker Group
Region
Recording Environment
Dialect Expression Tags
Dataset Split
Consent Metadata
```

## 구현 경계

최소한 다음 위치를 사용하거나  
현재 저장소 구조에 대응되는 위치를 확인한다.

```text
speech/surface_asr/
evaluation/asr/
data/benchmarks/
experiments/asr/
tests/surface_asr/
```

## 평가 지표

- CER
    
- Insertions
    
- Deletions
    
- Substitutions
    
- Dialect Expression Recall
    
- Surface Preservation Rate
    
- Context Overcorrection Rate
    
- Group-wise CER
    
- Environment-wise CER
    
- Confidence
    
- p50 Latency
    
- p95 Latency
    
- High-confidence Wrong Samples
    
- Low-confidence Correct Samples
    

## 산출물

```text
predictions.jsonl
metrics.json
error_samples.jsonl
high_confidence_errors.jsonl
report.md
experiment.yaml
```

## Audio Lab 통합

샘플별로 다음을 표시한다.

- 원본 음성
    
- Reference Surface Text
    
- Model Prediction
    
- Normalized Meaning
    
- CER
    
- 부산 표현 보존 여부
    
- 과보정 여부
    
- Confidence
    
- Latency
    
- Error Tags
    
- Model Version
    

## 통과 조건

- 고정 Benchmark 전체 추론 성공
    
- 오류 자동 Export
    
- 그룹별 지표 계산
    
- Audio Lab에서 결과 확인
    
- 동일 명령으로 재현 가능
    
- 아직 파인튜닝하지 않음
    

---

# 8. Gate 2: Surface ASR Improvement

Baseline 결과가 나온 뒤 다음 순서로 개선한다.

## 8.1 데이터와 레이블 검증

다음을 먼저 확인한다.

- 오디오와 전사가 일치하는가
    
- Surface Text가 실제 표현인가
    
- Normalized Text가 학습 정답에 섞이지 않았는가
    
- 오디오가 잘리거나 손상되지 않았는가
    
- 화자 중복이 없는가
    
- 같은 원본의 파생 조각이 Train/Test에 중복되지 않았는가
    

레이블 문제가 있으면 모델보다 레이블을 먼저 수정한다.

## 8.2 디코딩 실험

가중치 학습 전 다음을 비교한다.

```text
Greedy Decoding
Beam Search
Context Biasing
Dialect Lexicon Boosting
Language Model Weight
```

주요 변수는 실험당 하나만 바꾼다.

## 8.3 Tokenizer Audit

확인 항목:

- 부산 표현 출력 가능 여부
    
- UNK 발생
    
- 과도한 Subword 분할
    
- 종결형 분해
    
- 자모·음절·문자 단위의 장단점
    
- Token Frequency
    

## 8.4 CTC Head Fine-tuning

```text
Encoder Freeze
CTC Head만 학습
```

목적:

- 학습 Pipeline 검증
    
- 레이블과 Tokenizer 검증
    
- 빠른 첫 Checkpoint 생성
    

## 8.5 Partial Encoder Unfreeze

```text
CTC Head
+
FastConformer 상위 Layer 일부
```

## 8.6 Full Low-LR Fine-tuning

충분한 데이터와 안정된 Benchmark가 있을 때만 수행한다.

## 통과 조건

- Dialect Preservation 개선
    
- Context Overcorrection 감소
    
- 일반 한국어 성능의 심각한 붕괴 없음
    
- Test 화자에서도 개선
    
- High-confidence Wrong 감소
    
- 변경 원인 설명 가능
    

---

# 9. Gate 3: Streaming ASR

## 목표

사용자가 말하는 동안 낮은 지연으로 부분 전사를 생성한다.

## 후보 구조

```text
FastConformer
+
RNNT 또는 TDT
```

## 구현 요소

- Audio Chunk
    
- Streaming State
    
- Encoder Cache
    
- Partial Transcript
    
- Stable Prefix
    
- Unstable Suffix
    
- Endpointing
    
- Streaming Confidence
    
- Session Reset
    
- Cancellation
    

## 출력 예

```json
{
  "partial": true,
  "stable_prefix": "니 지금",
  "unstable_suffix": "어데",
  "stability": 0.81,
  "audio_end_ms": 1420
}
```

## Surface ASR과의 정책

```text
빠른 대화 반응
→ Streaming ASR

학습 기록과 정밀 분석
→ Surface ASR
```

두 결과가 의미적으로 크게 충돌하면  
Dialogue와 Teaching에 해당 불일치를 전달한다.

## 평가

- Partial Stability
    
- Final Agreement with Surface ASR
    
- Endpointing Accuracy
    
- First Partial Latency
    
- Final Latency
    
- Reset Correctness
    
- Memory Growth
    

---

# 10. Gate 4: Dialect-aware G2P and Target IPA

## 목표

부산 문장에서 발음 평가에 사용할  
검수 가능한 목표 IPA 후보를 생성한다.

## Pipeline

```text
Text Normalization
→ Morphological Analysis
→ Standard Korean G2P
→ Contextual Phonological Rules
→ Busan Dialect Rules
→ IPA
→ Allowed IPA Variants
```

## 생성할 자산

```text
ipa-inventory-v1.json
articulatory-features-v1.json
korean-g2p-rules-v1.yaml
busan-g2p-rules-v1.yaml
approved-expressions.jsonl
```

## 고려할 규칙

- 연음
    
- 비음화
    
- 유음화
    
- 경음화
    
- 구개음화
    
- 받침 중화
    
- ㅎ 탈락과 축약
    
- 자음군 단순화
    
- 음절 경계 재구성
    
- 부산 어휘와 종결형
    
- 세대별 변이
    
- 사투리 강도
    
- 표준어 혼용
    

## 승인 정책

AI가 생성한 IPA 후보는 기본적으로 `candidate`다.

부산 원어민 또는 언어 전문가가 승인한 후보만  
Production 평가 기준으로 사용한다.

---

# 11. Gate 5: Direct IPA Phonetic Model

## 목표

사용자가 의도한 단어가 아니라  
실제로 낸 음소를 IPA Sequence로 출력한다.

## 구조 후보

```text
SSL Speech Encoder 또는 FastConformer
+
CTC IPA Decoder
```

## 필수 출력

- IPA Sequence
    
- Token Confidence
    
- Start and End Time
    
- Frame Logits Reference
    
- Utterance Confidence
    

## 데이터 Pipeline

```text
Audio
→ Teacher Ensemble
→ Pseudo-label
→ G2P and IPA Candidate
→ Model Agreement
→ Human Review
→ Approved IPA Label
→ Training
```

Teacher 후보:

```text
Whisper 계열
SSL Phonetic Model
한국어 ASR + G2P
현재 Production Phonetic Model
```

## 학습 순서

```text
Encoder Freeze + IPA CTC Head
→ 상위 Encoder 일부 Unfreeze
→ 낮은 Learning Rate Fine-tuning
```

## 평가

- PER
    
- Phoneme Confusion Matrix
    
- Substitution
    
- Deletion
    
- Insertion
    
- IPA Preservation Rate
    
- Mispronunciation Recall
    
- False Correction Rate
    
- Group-wise Performance
    
- Human Agreement
    

---

# 12. Gate 6: Forced Alignment and Pronunciation Diagnosis

## 두 종류의 정렬

```text
Target IPA ↔ Observed IPA
Audio ↔ Target IPA
```

## 구현 후보

- Weighted Levenshtein
    
- CTC Segmentation
    
- Viterbi Alignment
    
- Posterior Alignment
    
- GOP-like Scoring
    

모든 음소 대치를 동일 비용으로 처리하지 않는다.

조음 특징을 이용한다.

- Place
    
- Manner
    
- Aspiration
    
- Tenseness
    
- Vowel Height
    
- Vowel Backness
    
- Rounding
    

## 출력 예

```json
{
  "segment": "구",
  "start_ms": 420,
  "end_ms": 550,
  "target_ipa": "k",
  "observed_ipa": "kʰ",
  "error_type": "aspiration_substitution",
  "severity": 0.71,
  "confidence": 0.88,
  "feedback_key": "reduce_aspiration"
}
```

## Confidence Gate

```text
높은 Confidence
→ 직접 교정 가능

중간 Confidence
→ 부드러운 제안

낮은 Confidence
→ 교정 보류 또는 재녹음
```

## 평가

- Alignment Accuracy
    
- Error Location Accuracy
    
- Error Type Accuracy
    
- False Positive Rate
    
- False Negative Rate
    
- Severity Correlation
    
- Calibration Error
    

---

# 13. Gate 7: Prosody and Dialect Acoustic Model

## Prosody 특징

- F0
    
- Pitch Range
    
- Pitch Slope
    
- Syllable Duration
    
- Vowel Duration
    
- Energy
    
- Pause
    
- Speech Rate
    
- Rhythm
    
- Phrase-final Contour
    
- Accent Location
    

절대 Hz를 직접 비교하지 않는다.

```text
log-F0
speaker normalization
semitone conversion
voiced-region interpolation
```

시간축은 음절 Alignment와 DTW로 정렬한다.

## 개발 순서

```text
Rule and Distance Baseline
→ Native Prosody Prototype
→ Small Classifier
→ Metric Learning or Sequence Model
```

## Dialect Acoustic 출력

```json
{
  "dialect_region": "busan_gyeongsang",
  "dialect_probability": 0.86,
  "dialect_strength": 0.62,
  "mixed_with_standard": true
}
```

## 필수 평가

- Speaker-disjoint
    
- Device-disjoint
    
- Environment-disjoint
    
- Gender and Age Groups
    
- Standard and Dialect Mixed Speech
    

특정 화자나 마이크를 방언으로 암기하지 않게 한다.

---

# 14. Gate 8: Speech Understanding Core

다음 결과를 하나의 공통 객체로 결합한다.

```text
Streaming ASR
Surface ASR
Direct IPA
Alignment
Pronunciation
Prosody
Dialect Acoustic
```

## 통합 출력 예

```json
{
  "utterance_id": "utt_001",
  "streaming_text": "니 지금 어데고",
  "surface_text": "니 지금 어데고",
  "normalized_meaning": "너 지금 어디야",
  "phonetic": {},
  "pronunciation": {},
  "prosody": {},
  "dialect_acoustic": {},
  "confidence": {
    "meaning": 0.94,
    "pronunciation": 0.81,
    "prosody": 0.72,
    "dialect": 0.86
  }
}
```

## Fusion

초기 후보:

- Logistic Regression
    
- Gradient Boosting
    
- Small MLP
    
- Isotonic or Temperature Calibration
    

## Shared Encoder 실험

독립 Baseline 확보 후에만 다음을 실험한다.

```text
Shared FastConformer Encoder
├─ RNNT/TDT
├─ CTC
├─ IPA Head
├─ Prosody Head
└─ Dialect Head
```

Shared Encoder가 독립 모델보다 실제 이득이 있을 때만 승격한다.

승격 기준:

- 핵심 지표 유지 또는 개선
    
- Latency 또는 GPU 비용 개선
    
- 각 Head의 회귀 없음
    
- 독립 모델 Rollback 가능
    

---

# 15. Parallel Track: Busan TTS

Audio Lab 완료 직후부터 TTS 준비를 병렬로 시작한다.

## TTS Data

- 깨끗한 부산 원어민 음성
    
- 일관된 마이크와 공간
    
- 정확한 Text와 IPA
    
- Relationship
    
- Emotion
    
- Dialect Strength
    
- Question / Statement / Exclamation
    
- Short Reactions
    
- Backchannels
    
- Conversation Context
    

## TTS Pipeline

```text
Text Normalization
→ Morphological Analysis
→ Dialect Lexicon
→ G2P
→ IPA
→ Prosody Plan
→ Acoustic / Speech-token Model
→ Vocoder / Codec Decoder
→ Audio
```

## 두 모드

### Conversation Mode

- 자연스러운 대화
    
- 축약
    
- 맞장구
    
- 감정
    
- 빠른 반응
    

### Reference Mode

- 또렷한 발음
    
- 음절과 억양 식별 가능
    
- 부산 억양 유지
    
- 반복 비교 가능
    

## 조건

- Speaker
    
- Emotion
    
- Relationship
    
- Dialect Strength
    
- Speech Rate
    
- Pitch Style
    
- Pause
    
- Emphasis
    

## 오류 분리

- Text Normalization
    
- G2P
    
- IPA
    
- Duration
    
- Pitch
    
- Acoustic Model
    
- Vocoder
    
- Streaming
    

## 평가

- MOS
    
- Busan Naturalness
    
- Pronunciation Accuracy
    
- Prosody Fit
    
- Speaker Consistency
    
- Relationship Fit
    
- First-chunk Latency
    
- Real-time Factor
    
- Long-form Stability
    

Audio Lab에 TTS A/B 평가 화면을 추가한다.

---

# 16. Gate 9: Busan Domain Brain

## Dialect Knowledge Base

표현을 구조화하여 저장한다.

```json
{
  "expression_id": "eodego",
  "surface_forms": ["어데고", "어데 있노"],
  "standard_meaning": "어디야",
  "relationship": ["close_friend"],
  "formality": "casual",
  "dialect_strength": 2,
  "allowed_ipa_variants": [],
  "prosody_patterns": [],
  "examples": [],
  "anti_examples": [],
  "approval_status": "approved"
}
```

검색은 다음을 결합한다.

```text
Metadata Filter
+
Lexical Search
+
Embedding Search
```

## Dialect Understanding

입력:

```text
Surface Form
Normalized Meaning
IPA and Prosody
Relationship
Scenario
Knowledge Base
```

출력:

```json
{
  "meaning_passed": true,
  "relationship_fit": true,
  "dialect_naturalness": 0.72,
  "recommended_forms": [
    "니 지금 어데고?",
    "지금 어데 있노?"
  ]
}
```

LLM은 후보 분석을 생성한다.

규칙과 지식베이스가 검증한다.

---

# 17. Gate 10: Tutor Coordinator

## 구성

```text
Scenario Engine
Learner Model
Teaching Policy
Curriculum
Dialect Knowledge
Confidence Gate
```

## Teaching Actions

```text
NO_CORRECTION
NATURAL_RECAST
SOFT_TIP
EXPLICIT_CORRECTION
SEGMENT_RETRY
REFERENCE_COMPARISON
CLARIFICATION
```

## 기본 정책

```text
의미 실패
→ CLARIFICATION

첫 작은 오류
→ NATURAL_RECAST

반복 음소 오류
→ EXPLICIT_CORRECTION 또는 SEGMENT_RETRY

억양 오류
→ REFERENCE_COMPARISON

관계 오류
→ SOFT_TIP

Confidence 낮음
→ NO_CORRECTION
```

## Learner Model

최소 저장 상태:

- Expression Comprehension
    
- Expression Production
    
- Phoneme Weakness
    
- Prosody Weakness
    
- Relationship Skill
    
- Hint Dependency
    
- Retry Success
    
- Last Practice
    
- Repeated Errors
    
- Forgetting Estimate
    

초기에는 Rule, Elo 또는 Bayesian Knowledge Tracing을 사용할 수 있다.

---

# 18. Gate 11: Dialogue Planner and Structured Speech Plan

## Dialogue Planner 입력

- Conversation History
    
- Character
    
- Relationship
    
- Scenario
    
- Learner State
    
- Teaching Action
    
- Retrieved Dialect Knowledge
    
- Allowed Expressions
    
- Disallowed Expressions
    
- Speech Constraints
    

## 출력

```json
{
  "response_text": "알았다. 쪼매만 기다리면 되는 거제?",
  "normalized_meaning": "알겠어. 조금만 기다리면 되는 거지?",
  "teaching_strategy": "natural_recast",
  "emotion": "friendly",
  "dialect_strength": 2
}
```

## Response Validation

- 의미 일치
    
- 관계 적합성
    
- 검수된 부산 표현
    
- 과도한 방언 사용 여부
    
- 사용자 수준
    
- 금지 내용
    
- Schema 유효성
    

## Structured Speech Plan

```json
{
  "text": "알았다. 쪼매만 기다리면 되는 거제?",
  "ipa": [],
  "speaker_id": "busan_friend_01",
  "emotion": "friendly",
  "relationship": "close_friend",
  "dialect_strength": 2,
  "speech_rate": 1.02,
  "pitch_style": "busan_casual_confirmation",
  "duration_controls": {},
  "pause_plan": [],
  "emphasis": ["쪼매만"],
  "mode": "conversation"
}
```

이 객체는 Dialogue와 TTS 사이의 고정 계약이다.

---

# 19. Gate 12: Busan TTS Integration

Structured Speech Plan을 입력으로 받아 실제 음성을 생성한다.

## 필수 기능

- Conversation Mode
    
- Reference Mode
    
- Speaker Consistency
    
- Emotion Control
    
- Dialect Strength Control
    
- Speech Rate Control
    
- Prosody Style
    
- Emphasis
    
- Pause
    
- Streaming Chunk
    
- Cancellation
    

## 완료 조건

같은 문장을 다음 조건으로 다르게 생성한다.

```text
친구에게 자연스럽게
어른에게 공손하게
장난스럽게
걱정스럽게
학습용으로 또렷하게
```

---

# 20. Gate 13: Offline Vertical Slice

이 Gate는 부산 특화 모듈들이 성숙한 뒤 검증하는 정식 Offline Vertical Slice다.
Track A의 `Offline Vertical Slice v0`는 이 Gate를 조기에 통과했다고 선언하기 위한 것이 아니며, 기존 Provider와 단순 로직으로 시스템 계약과 사용자 흐름을 먼저 검증하는 별도 제품 마일스톤이다.

실시간화 전에 다음 파일 기반 정식 경로를 완성한다.

```text
WAV
→ Surface ASR
→ Direct IPA
→ Alignment
→ Pronunciation
→ Prosody
→ Dialect Understanding
→ Tutor Coordinator
→ Dialogue Planner
→ Speech Plan
→ Busan TTS
→ Output WAV
→ Learner Model Update
```

## 완료 예

사용자:

```text
나 버스 놓쳤어. 조금만 기다려.
```

시스템:

```text
의미 성공
부산 표현 없음
발음 오류 없음
표준어 Prosody
첫 번째 작은 오류
NATURAL_RECAST
```

AI 출력:

```text
알았다. 쪼매만 기다리면 되는 거제?
```

UI:

```text
조금만 → 쪼매만
```

Offline Vertical Slice가 안정적이지 않은 상태에서  
최종 WebRTC 통합을 진행하지 않는다.

---

# 21. Gate 14: Real-time Integration

## 빠른 경로

```text
Streaming ASR
→ Dialogue Planner
→ Streaming TTS
```

## 정밀 경로

```text
Surface ASR
→ IPA
→ Alignment
→ Pronunciation
→ Prosody
→ Teaching Policy
```

## 장기 경로

```text
Session
→ Learner Model
→ Review
→ Training Data Candidate
```

## 구현 요소

- WebRTC
    
- VAD
    
- AEC
    
- Noise Suppression
    
- Jitter Buffer
    
- Barge-in
    
- TTS Cancellation
    
- Timeout
    
- Retry
    
- Idempotency
    
- Backpressure
    
- Session Recovery
    
- Event Versioning
    

## 핵심 이벤트

```text
speech.started
speech.partial
speech.ended
asr.finalized
phonetic.completed
assessment.completed
teaching.decided
response.planned
tts.started
tts.completed
learner.updated
```

정밀 분석이 늦어도 대화 전체를 멈추지 않는다.

---

# 22. Gate 15: Data Flywheel and Production

## Data Flywheel

```text
실사용 음성
→ Consent Check
→ Failure Mining
→ Low-confidence Mining
→ Active Learning Queue
→ Human Review
→ Label Update
→ Retraining
→ Fixed Benchmark
→ Staging
→ Production
```

## 필수 운영 시스템

- Dataset Registry
    
- Model Registry
    
- Experiment Tracking
    
- Admin Review
    
- Monitoring
    
- Distributed Tracing
    
- Consent Management
    
- Deployment
    
- Rollback
    
- Cost Tracking
    

## 실험 재현 요소

```text
Code Version
Dataset Version
Config
Container
Random Seed
Checkpoint
Metrics
Hardware
```

---

# 23. 사용자에게 병렬로 가르칠 내용

각 Gate를 구현한 뒤 다음 형식으로 설명한다.

```text
이번 Gate에서 만든 것
최종 아키텍처에서의 위치
입력과 출력
핵심 코드 경로
모델 구조
정답 레이블
Loss
평가 지표
실패 사례
사용자가 반드시 이해할 개념
지금 몰라도 되는 개념
다음 Gate
```

사용자가 모든 이론을 선행 학습하도록 요구하지 않는다.

해당 Gate를 디버깅할 수 있는 수준까지만 설명한다.

---

# 24. AI 에이전트 병렬 운영

병렬 Agent 또는 Git Worktree 기능이 있다면 다음 소유권을 사용한다.

## ASR Agent

```text
speech/surface_asr/
speech/streaming_asr/
evaluation/asr/
```

## Phonetic Agent

```text
speech/ipa_recognizer/
speech/forced_alignment/
speech/pronunciation/
knowledge/ipa/
```

## Prosody Agent

```text
speech/prosody/
speech/dialect_acoustic/
evaluation/prosody/
```

## TTS Agent

```text
speech/tts/
knowledge/g2p/
evaluation/tts/
```

## Tutor Agent

```text
services/tutor_coordinator/
services/scenario_engine/
services/learner_model/
services/dialogue_planner/
knowledge/expressions/
```

## Platform Agent

```text
apps/
services/orchestrator/
gateways/
infra/
```

## Reviewer Agent

다음을 검사한다.

- Data Leakage
    
- Metric Bugs
    
- Schema Violations
    
- Fake Results
    
- Missing Tests
    
- Regression
    
- Security
    
- Reproducibility
    

같은 파일을 여러 Agent가 동시에 수정하지 않게 하라.

병렬 Agent가 없다면 이 소유권을 논리적 작업 경계로 유지한다.

---

# 25. 모든 Gate의 보고 형식

각 Gate 완료 후 다음 형식으로 보고한다.

```text
Gate:
현재 목표:
아키텍처 위치:
변경 파일:
실행 명령:
테스트 결과:
Benchmark 결과:
개선된 항목:
악화된 항목:
높은 Confidence 오답:
실패 원인:
불확실한 항목:
사용자가 이해해야 할 개념:
다음 Gate:
```

성공하지 않은 Gate를 완료라고 표시하지 마라.

실행하지 못한 명령은 실행했다고 주장하지 마라.

---

# 26. 현재 즉시 실행할 Task

현재 즉시 실행할 Task는 다음과 같다.

```text
TASK-004
Busan ASR Training Dataset Foundation
```

Phase A의 immutable Manifest, train/validation 분리, Benchmark·audio lineage
누출 검사와 모델 중립 export 계약은 구현됐다. 현재 즉시 실행할 부분은 이
계약에 들어갈 새 학습 동의 발화 300~500개의 수집과 Surface label 사람
검수다. 실제 데이터가 없으므로 TASK-004 전체와 Fine-tuning은 완료로
표시하지 않는다.

## 수행 순서

1. `busan-surface-v0@1.0.0`을 test-only Benchmark로 잠그고 학습 금지 확인
    
2. 별도 train/validation Utterance와 Manifest 계약 정의
    
3. Benchmark와 speaker/audio lineage가 겹치지 않도록 검증
    
4. 원본·파생 WAV와 transcript의 provenance·hash 보존
    
5. Surface transcript와 방언 표현 라벨 검수 상태 정의
    
6. train/validation split의 화자·표현·환경 분포 기록
    
7. 중복 음성·중복 transcript·Benchmark 누출 자동 검사
    
8. 첫 300~500개 Pilot 발화 수집 워크플로 준비
    
9. 소규모 학습 파이프라인 smoke test용 export 계약 준비
    
10. 데이터가 준비된 뒤 TASK-005 모델 후보를 다시 선택
    

## 아직 하지 않을 것

- Benchmark를 이용한 파인튜닝
    
- Test 화자 학습
    
- Shared Encoder 공동학습
    
- WebRTC 최종 통합
    
- 전체 시스템 재작성
    
- 측정 결과 없는 성능 주장

- IPA
    
- Forced Alignment
    
- Prosody scoring
    
- Speech Understanding Core
    
- Learner Model
    
- Dialogue Planner 전체 구현
    
- 부산 TTS 학습
    
- Streaming
    
- ASR fine-tuning
    
- Benchmark 확장
    

---

# 27. 첫 응답 형식

이 스크립트를 받은 즉시 다음 형식으로 시작하라.

```text
현재 Phase:
현재 Task:
저장소 조사 계획:
Audio Lab 예상 진입점:
확인할 Schema:
확인할 Benchmark:
가장 위험한 가정:
첫 검증 실험:
성공 기준:
수정 예정 파일:
실행 예정 명령:
```

그다음 실제 저장소를 확인하고 구현으로 이동하라.

추상적인 비전 설명을 반복하지 마라.

저장소를 확인하지 않은 상태에서  
존재하지 않는 파일이나 결과를 만들어내지 마라.

이제 `TASK-004`를 시작하라.

이 스크립트는 **Master Identity 다음에 붙이는 실행 지침**이야. Audio Lab 완성 뒤 Codex가 어디부터 시작하고, 어떤 Gate로 확장하고, 무엇을 측정하며, 언제 최종 실시간 시스템으로 통합할지를 한 번에 고정한다.
