
최종 제품은 네가 말한 순서로 실행돼.

```text
WebRTC
→ 실시간 미디어 처리
→ Shared Speech Encoder
→ RNNT/TDT · CTC · 음성 분석
→ Dialect Understanding
→ Tutor Coordinator
→ Dialogue Planner
→ Speech Plan
→ 부산 TTS
→ 사용자에게 음성 출력
```

하지만 개발할 때는 WebRTC부터 순서대로 만드는 게 아니야. **핵심 모델을 측정할 수 있는 환경부터 만든 뒤, 위험도가 높은 모델을 먼저 검증하고 마지막에 실시간으로 연결**해야 해.

그리고 Audio Lab은 AI 모델 대신 만드는 프로그램이 아니야.

> **Audio Lab은 AI 모델을 반복해서 테스트하고 비교하기 위한 실험 장비다.**

---

# 1. Audio Lab을 왜 만드는가

예를 들어 FastConformer-CTC 모델을 그냥 실행하면 터미널에 이런 결과만 나올 수 있어.

```text
국밥 하나 주세요
```

그런데 이 결과만 보고는 다음을 알기 어려워.

```text
원래 음성이 무엇이었는가?
정답 전사는 무엇이었는가?
모델이 어느 글자를 틀렸는가?
“주이소”를 “주세요”로 고친 것인가?
음질이 나빠서 틀린 것인가?
모델이 높은 확신으로 틀린 것인가?
이전 모델보다 개선된 것인가?
```

Audio Lab은 이것을 한 화면에서 보게 해.

```text
원본 음성 재생
정답 Surface transcript
모델 예측 transcript
CER
부산 표현 보존 여부
표준어 과보정 여부
모델 confidence
파형
Mel spectrogram
F0
모델 버전
```

예를 들면:

```text
원본 음성
“국밥 하나 주이소”

정답
“국밥 하나 주이소”

모델 예측
“국밥 하나 주세요”

판정
Dialect normalization error

부산 표현 보존
실패

Confidence
0.94
```

이 샘플은 특히 위험해. 모델이 틀렸는데 확신까지 높기 때문이야.

## Audio Lab은 모델 테스트 장치다

```text
FastConformer-CTC v0
          ↓
       Audio Lab
          ↓
오류 확인·지표 측정

FastConformer-CTC v1
          ↓
       Audio Lab
          ↓
v0와 v1 비교
```

그러니까:

```text
Audio Lab 또는 AI 모델
```

중 하나를 고르는 게 아니야.

```text
AI 모델
+
Audio Lab에서 테스트
```

가 맞아.

로켓으로 비유하면 Audio Lab은 로켓이 아니라 **엔진 연소 시험대와 센서 시스템**이야. 엔진만 켰다가 꺼지면 “실패했다”밖에 모르지만, 센서가 있으면 압력, 온도, 진동, 연료 흐름 중 무엇이 문제였는지 알 수 있지.

---

# 2. Audio Lab에서 테스트하는 것

Audio Lab은 최종적으로 네 모든 음성 모델을 테스트하는 공통 연구 화면이 돼.

## 첫 단계: 오디오 자체 검사

```text
Sample rate
채널 수
음성 길이
Clipping
무음 비율
음량
파형
Mel spectrogram
F0
```

이유는 모델이 틀린 것과 입력 음성이 잘못된 것을 구분하기 위해서야.

예:

```text
모델 오류처럼 보임
실제 원인: 48kHz 음성을 16kHz로 잘못 변환

발음 오류처럼 보임
실제 원인: 노이즈 제거가 ㅋ의 기식 구간을 훼손
```

## 두 번째: Surface ASR 테스트

```text
정답 문장
모델 전사
CER
부산 표현 보존율
표준어 과보정률
```

## 세 번째: IPA 모델 테스트

나중에는 같은 화면에 추가해.

```text
목표 IPA
사용자 관찰 IPA
PER
음소 대치·삭제·삽입
토큰 confidence
```

## 네 번째: Forced Alignment

```text
목표 음소
사용자 음소
음소별 시작·종료 시간
오류 구간 재생
```

## 다섯 번째: Prosody

```text
기준 F0
사용자 F0
음절별 길이
문장 끝 억양
리듬 차이
```

## 여섯 번째: TTS

```text
같은 문장의 TTS v1 / v2 A·B 비교
부산 자연스러움
발음 정확성
화자 일관성
억양 적절성
```

그래서 Audio Lab은 초반에 잠깐 만들고 버리는 페이지가 아니야.

> **ASR, IPA, Prosody, TTS의 모든 실험 결과가 모이는 네 연구소의 계기판**이야.

---

# 3. 지금 ASR부터 만드는 게 맞는가

맞아. 정확히는 **Speech Understanding Core의 첫 번째 검증 대상으로 Surface ASR부터 시작**하는 거야.

그렇다고 최종 아키텍처가 ASR 중심이라는 뜻은 아니야. TTS도 병렬로 시작해야 해.

Surface ASR을 첫 모델로 선택하는 이유는 세 가지야.

## 이유 1. 현재 음성 모델의 실패를 가장 빨리 볼 수 있다

```text
음성
→ 사전학습 CTC 모델
→ 전사 결과
→ 정답과 비교
```

ASR은 비교적 빠르게 baseline을 만들 수 있어.

## 이유 2. 이후 데이터 정제에 필요하다

```text
음성
→ Surface ASR
→ 실제 표현 후보
→ 검수
→ ASR·IPA 학습 데이터
```

## 이유 3. 정밀 분석 경로의 기준이 된다

```text
Surface ASR
→ 목표 문장 후보
→ G2P
→ 목표 IPA
→ IPA Alignment
→ 발음 평가
```

Surface ASR이 모든 판단을 독점하는 건 아니지만, 다음 단계들이 시작할 문장 기준을 제공해.

---

# 4. 최종 아키텍처와 실제 개발 순서를 구분하자

## 최종 실행 아키텍처

사용자가 실제로 앱을 사용할 때는 이 순서야.

```text
사용자 마이크
    ↓
WebRTC
    ↓
실시간 미디어 처리
VAD · AEC · Noise Suppression · Buffer
    ↓
Shared Speech Encoder
FastConformer
    ├─ RNNT/TDT
    │  실시간 부분 전사
    │
    ├─ CTC
    │  Surface 정밀 전사·정렬
    │
    ├─ IPA Phonetic Branch
    │  실제 음소 인식
    │
    ├─ Prosody Branch
    │  F0·Duration·Rhythm
    │
    └─ Dialect Acoustic Branch
       부산 음향 특징·강도
    ↓
Dialect Understanding
    ↓
Tutor Coordinator
    ↓
Dialogue Planner + 교육 개입
    ↓
Structured Speech Plan
    ↓
Dialect Speech Generator
    ↓
부산 사투리 음성
```

네가 정리한 아키텍처는 이 방향과 일치해.

## 실제 개발 순서

```text
측정 환경
→ 독립 모델 baseline
→ 각 모델 평가
→ 음성 분석 통합
→ 교사 두뇌
→ TTS 통합
→ 실시간 연결
→ 운영·재학습
```

WebRTC부터 만드는 게 아닌 이유는, 실시간 입력이 들어와도 모델이 맞게 듣는지 평가할 수 없다면 배관만 만든 셈이기 때문이야.

## 실행 전략: 제품 검증과 핵심 모델 연구를 병행한다

위 순서는 최종 모듈의 기술 의존성과 연구 성숙도 순서로 유지한다.
그러나 모든 모델이 완성될 때까지 제품 통합을 미루지는 않는다.

현재부터 다음 두 트랙을 병행한다.

### Track A: 제품 Offline Vertical Slice

```text
사용자 WAV
→ 기존 ASR Provider
→ 단순 Surface·방언 표현 평가
→ 구조화된 LLM Tutor 판단
→ 기존 또는 가볍게 적용한 TTS Provider
→ 부산 응답 WAV
```

이 경로는 최종 모델 품질을 증명하는 경로가 아니다.
전체 학습 경험, 모듈 간 계약, 오류 전달 방식과 WAV 입출력을 조기에 검증하는 `Offline Vertical Slice v0`다.

최소 계약은 다음과 같다.

```text
ASRProvider
SimpleAssessmentService
TutorProvider
TTSProvider
OfflineSpeechPipeline
```

초기 구현은 사전학습 모델, 외부 API 또는 단순 규칙을 사용할 수 있다.
부산 특화 모델이 준비되면 호출부를 바꾸지 않고 Provider 내부 구현을 교체한다.

### Track B: 핵심 모델 연구

```text
Audio Lab
→ 고정 Benchmark
→ 독립 모델 후보 실험
→ 정량 평가와 사람 검수
→ 실패 유형별 개선
```

`TASK-002 NVIDIA RIVA Korean Conformer-CTC Surface ASR Baseline`과
`TASK-003A Surface ASR Evaluation Calibration`, `TASK-003B Nemotron 3.5
Pretrained Surface ASR Baseline Integration`과 `TASK-003C Nemotron Inference
Configuration Validation`은 완료됐다. 현재 Track B의 작업은 `TASK-004 Busan
ASR Training Dataset Foundation`이다. 고정된 10개 Pilot Benchmark는 평가에만
사용하고 학습 데이터로 사용하지 않는다. TASK-004 Phase A의 Manifest,
split·누출 검증과 모델 중립 export 계약은 구현됐으며, 실제 새 학습 발화
300~500개 수집은 아직 시작 전이다.

```text
TASK-002 RIVA Korean Conformer-CTC Baseline                         완료
TASK-003A Surface ASR Evaluation Calibration                       완료
TASK-003B Nemotron 3.5 Pretrained Surface ASR Baseline Integration 완료(제한 있음)
TASK-003C Nemotron Inference Configuration Validation              완료
TASK-004 Busan ASR Training Dataset Foundation                     진행(Phase A 완료)
TASK-005 Nemotron Busan Fine-tuning Pilot                          후보 보류
```

Whisper Cross-model Comparison은 삭제하지 않고 현재 우선순위에서 보류한다.

다음 연구 Gate는 번호만 보고 자동 선택하지 않는다.
Offline Vertical Slice와 사용자 테스트에서 가장 큰 실패를 기준으로 ASR, IPA·Alignment, Prosody·TTS, Teaching Policy 또는 Streaming 중 다음 우선순위를 정한다.

---

# 5. 실제 개발의 시작점과 끝점

전체 개발을 8단계로 보면 가장 명확해.

# 단계 0. 측정 기반

## 만드는 것

```text
공통 Utterance Schema
Audio Lab
Benchmark
오류 taxonomy
평가 코드
```

## 여기서 검증하는 것

```text
오디오가 정상인가?
정답 레이블이 올바른가?
모델 v0와 v1을 동일 조건에서 비교할 수 있는가?
```

## 완료 상태

```text
WAV 업로드
→ 오디오 검사
→ 모델 실행
→ 정답과 비교
→ 지표·오류 저장
```

이게 없으면 이후 모든 실험이 눈대중이 돼.

---

# 단계 1. Surface ASR

## 만드는 것

```text
Pretrained Korean Surface ASR baseline
→ 부산 데이터 파인튜닝
→ Surface transcript
```

## 평가

```text
CER
부산 표현 보존율
표준어 과보정률
그룹별 성능
Confidence
```

## 완료 상태

```text
“국밥 하나 주이소”
→ “국밥 하나 주이소”
```

처럼 실제 표현을 보존하고, 실패 사례가 자동으로 분류돼.

---

# 단계 2. IPA 발음 인식과 정렬

## 만드는 것

```text
Dialect-aware G2P
→ 목표 IPA 후보

Direct IPA Phonetic Model
→ 사용자 실제 IPA

Forced Alignment
→ 목표와 관찰 IPA 정렬
```

## 평가

```text
PER
대치·삭제·삽입
시간 구간
실제 오류 보존
False correction
```

## 완료 상태

```text
목표: k
사용자: kʰ

첫 음절 0.42~0.55초
기식이 강함
confidence 0.88
```

---

# 단계 3. Prosody와 Dialect Acoustic

## 만드는 것

```text
F0
Duration
Energy
Pause
Rhythm
Phrase-final contour
Dialect classification
Dialect strength
```

## 완료 상태

다음 두 경우를 구분할 수 있어야 해.

```text
부산 표현 + 부산 억양
부산 표현 + 표준어 억양
```

예:

```json
{
  "lexical_dialect_score": 0.91,
  "acoustic_dialect_score": 0.42,
  "mixed_with_standard": true
}
```

---

# 단계 4. Speech Assessment 통합

지금까지 만든 결과를 합쳐.

```text
Surface ASR
+
IPA
+
Alignment
+
GOP
+
Prosody
+
Dialect Acoustic
→ 최종 음성 평가
```

## 완료 상태

```json
{
  "meaning_confidence": 0.93,
  "pronunciation_status": "minor_issue",
  "prosody_status": "standard_like",
  "dialect_strength": 0.48,
  "correction_confidence": 0.86
}
```

이 단계가 Speech Understanding Core의 완성점이야.

---

# 단계 5. Dialect Understanding와 Tutor Brain

## 만드는 것

```text
Surface form
Normalized meaning
Recommended Busan form
관계·격식 분석
Dialect Knowledge Base
Teaching Policy
Learner Model
Scenario Engine
```

## 완료 상태

```text
사용자가 무엇을 의미했는가?
부산 표현으로 자연스러운가?
관계에 적절한가?
지금 교정할 것인가?
어떤 방식으로 교정할 것인가?
```

를 결정할 수 있어야 해.

---

# 단계 6. Dialogue Planner와 부산 TTS

## 만드는 것

```text
Dialogue Planner
→ 부산 응답 내용

Structured Speech Plan
→ 감정·강도·속도·강조·억양

Dialect Speech Generator
→ 실제 음성
```

## 완료 상태

같은 문장을 조건에 따라 다르게 말해.

```text
친구에게 장난스럽게
어른에게 공손하게
학습용으로 또렷하게
대화용으로 자연스럽게
```

---

# 단계 7. Streaming ASR과 실시간 통합

Surface ASR이 어느 정도 측정 가능한 상태가 되면 Streaming ASR은 병렬로 개발할 수 있어.

## 만드는 것

```text
FastConformer-RNNT/TDT
Chunk processing
Partial transcript
Stable prefix
Endpointing
Barge-in
Streaming TTS
```

## 실시간 경로

```text
빠른 경로
RNNT/TDT
→ Dialogue
→ TTS

정밀 경로
CTC
→ IPA
→ Prosody
→ Teaching

장기 경로
Session
→ Learner Model
→ 복습·재학습 데이터
```

이 세 경로를 분리하는 구조도 네 최종 설계에 포함되어 있어.

---

# 단계 8. 운영과 지속 학습

여기가 기술적으로 진짜 마지막이야.

```text
실사용 음성
→ 낮은 confidence·실패 샘플 수집
→ 개인정보·동의 검사
→ 사람 검수
→ 데이터 버전 갱신
→ 모델 재학습
→ 고정 Benchmark
→ 새 모델 배포
→ Rollback 가능
```

AI가 한 번 만들어지고 끝나는 게 아니라 계속 개선되는 루프가 완성돼.

---

# 6. 세 전문 도메인 트랙은 병렬로 진행해야 한다

완전히 순차적으로 기다릴 필요는 없어.

이 절의 세 트랙은 앞 절의 제품 Track A와 연구 Track B를 대체하지 않는다.
듣기, 말하기, 교육 지능을 나눈 장기 전문 도메인 분류다.

## 전문 영역 1: 듣는 귀

```text
Audio Lab
→ Surface ASR
→ IPA
→ Alignment
→ Prosody
→ Speech Assessment
```

## 전문 영역 2: 말하는 입

```text
부산 화자 데이터
→ Text normalization
→ G2P·IPA
→ TTS baseline
→ Prosody conditioning
→ Streaming TTS
```

## 전문 영역 3: 교사의 두뇌

```text
공통 Schema
→ Dialect Knowledge Base
→ Scenario
→ Teaching Policy
→ Learner Model
→ Dialogue Planner
```

단, 첫 통합 순서는 다음이야.

```text
듣는 귀가 기본 결과 출력
+
말하는 입이 부산 음성 생성
+
교사 두뇌가 규칙으로 중간 연결
→ Offline Vertical Slice
```

그다음 WebRTC를 붙여 실시간화해.

---

# 7. Shared Speech Encoder는 언제 만드는가

여기서 아주 중요한 정리가 필요해.

네 **최종 이상향**은:

```text
Shared FastConformer Encoder
├─ RNNT/TDT
├─ CTC
├─ IPA Head
├─ Prosody Head
└─ Dialect Head
```

야.

하지만 첫 개발부터 다섯 Head를 동시에 공동학습하면 다음 문제가 생겨.

```text
ASR Loss가 IPA 학습을 방해
IPA Loss가 ASR 성능을 악화
Prosody 데이터와 ASR 데이터 수량 불일치
어떤 Head 때문에 Encoder가 바뀌었는지 추적 어려움
```

따라서 개발은 이렇게 해야 해.

## 먼저 독립 baseline

```text
CTC ASR 모델
IPA 모델
Prosody 분석기
Dialect 분류기
```

각자 제대로 작동하고 평가되는지 확인해.

## 그다음 Encoder 공유 실험

```text
독립 baseline 성능
vs
Shared Encoder multi-task 성능
```

을 비교해.

즉:

```text
최종 아키텍처는 Shared Encoder

개발 첫 단계는 독립 모델 baseline
```

이야.

최종 설계와 최초 구현을 똑같이 만들려고 하면 디버깅 난도가 폭발해.

---

# 8. 지금 네가 실제로 시작할 부분

현재 작업은 정확히 이것이야.

```text
Track B
TASK-004 Busan ASR Training Dataset Foundation
```

오늘 기준 실행 순서:

```text
1. frozen Benchmark 10개 학습 금지 확인
2. 별도 train/validation 계약 정의
3. 화자와 audio lineage 분리
4. 원본·파생 음성 hash와 provenance 보존
5. Surface transcript와 방언 라벨 검수
6. split·중복·Benchmark 누출 검증
7. 300~500개 Pilot 수집 워크플로 준비
8. 학습 export 계약 후 모델 후보 재판정
```

TASK-002 Pilot은 다음 상태로 완료됐다.

```text
Experiment: task-002-nvidia-korean-conformer-ctc-pretrained-v0
Model: NVIDIA RIVA Conformer ASR Korean
Gate 1: baseline_established_with_limitations
```

현재 `TASK-004`에서는 다음을 구현하지 않는다.

```text
Nemotron fine-tuning
실제 300~500개 Training Dataset 수집
IPA
Forced Alignment
Prosody scoring
Speech Understanding Core
Learner Model
Dialogue Planner 전체
부산 TTS 학습
WebRTC·Streaming
Benchmark 확장
```

기존 RIVA와 Nemotron 결과는 삭제하거나 대체하지 않는다. TASK-003C에서
Nemotron 빈 출력은 실제 blank-only 모델 출력으로 확인됐다. Nemotron은 최종
모델로 확정되지 않았으며 TASK-005 대상은 데이터 기반 준비 후 다시 선택한다.
Phase A 구현 뒤의 다음 작업은 frozen Benchmark와 분리된 새 발화를 명시적
학습 동의와 함께 수집하고 Surface label을 사람 검수하는 것이다.

# 최종적으로 어디가 끝인가

한 번의 사용자 발화 기준으로 보면 끝은 다음 세 결과가 모두 저장되는 시점이야.

```text
1. 사용자에게 부산 음성이 재생됨
2. 필요한 Tip·재시도가 화면에 표시됨
3. Learner Model이 업데이트됨
```

전체 시스템 개발 기준으로는 여기까지야.

```text
사용자 음성
→ 실시간 대화
→ 정밀 음성 평가
→ 교육 판단
→ 부산 음성 생성
→ 학습 상태 업데이트
→ 실패 데이터 검수
→ 재학습·재배포
```

즉, 마지막은 단순히 TTS WAV 파일이 나오는 순간이 아니야.

> **사용자 경험이 끝나고, 그 결과가 다음 학습과 모델 개선으로 다시 연결되는 순간**이 최종 아키텍처의 끝이야.

지금은 그 거대한 고리 중 첫 번째 계측 지점인 **Surface ASR Baseline과 평가 장치**부터 만드는 단계다.
