
> **사용자의 실제 발화를 정밀하게 듣고, 부산 사투리 표현·발음·억양을 평가하며, 상황에 맞는 부산 사람처럼 직접 말하고, 필요한 순간에만 자연스럽게 가르치는 실시간 AI 튜터 시스템**

---

# 1. 최종 목표 구조

겉으로는 부산 현지인 캐릭터 한 명과 영상통화하는 것처럼 보여야 한다.

하지만 내부에서는 다음 여섯 개의 지능이 함께 작동한다.

```text
1. 듣는 귀
   사용자가 실제로 무엇을 말했는지 보존

2. 음성 분석기
   발음·받침·억양·길이·리듬 분석

3. 사투리 언어학자
   표준어 의미와 부산 표현의 차이 판단

4. 부산 캐릭터
   관계·상황에 맞게 부산 사투리로 반응

5. 숨은 교사
   언제, 무엇을, 어떤 방식으로 고칠지 결정

6. 학습자 기억
   사용자가 배운 것과 반복하는 실수를 저장
```

최종 서비스 흐름은 다음과 같다.

```text
사용자 음성
    ↓
실시간 음성 인식·분석
    ↓
의미·사투리·발음·억양 해석
    ↓
상황 미션과 사용자 실력 확인
    ↓
대화 반응 + 교육 개입 결정
    ↓
부산 사투리 음성 생성
    ↓
캐릭터 음성 응답 + 화면 Tip
```

---

# 2. 최종 프로덕션 아키텍처

```text
┌─────────────────────────────────────┐
│             모바일 앱                │
│ 카메라 · 마이크 · 스피커 · 자막 · UI │
└──────────────────┬──────────────────┘
                   │ WebRTC
                   ▼
┌─────────────────────────────────────┐
│       실시간 미디어 처리 계층         │
│ VAD · 에코 제거 · 잡음 억제 · 버퍼링  │
└──────────────────┬──────────────────┘
                   ▼
┌─────────────────────────────────────┐
│        Shared Speech Encoder         │
│           FastConformer              │
└───────┬─────────┬─────────┬─────────┘
        │         │         │
        ▼         ▼         ▼
  RNNT/TDT      CTC       음성 분석 Head
  실시간 전사   정밀 전사  음소·억양·방언
        │       및 정렬     특징
        └─────────┬─────────┘
                  ▼
┌─────────────────────────────────────┐
│       Dialect Understanding          │
│ 실제 발화 · 표준어 의미 · 부산 표현  │
│ 관계 적절성 · 사투리 강도 · 자연스러움│
└──────────────────┬──────────────────┘
                   ▼
┌─────────────────────────────────────┐
│          Tutor Coordinator           │
│                                     │
│ Scenario Engine                     │
│ Learner Model                       │
│ Teaching Policy                     │
│ Dialect Knowledge Base              │
└────────────┬──────────────┬─────────┘
             │              │
             ▼              ▼
     Dialogue Planner   교육 개입 결정
     캐릭터 반응 계획   Tip·재시도·복습
             └──────────────┬─────────┘
                            ▼
┌─────────────────────────────────────┐
│       Dialect Speech Generator       │
│ 부산 문장 · 억양 · 감정 · 속도 · 음색│
└──────────────────┬──────────────────┘
                   ▼
          실제 부산 사투리 음성
```

NeMo는 하나의 인코더 위에 RNNT와 CTC 디코더를 함께 두는 하이브리드 ASR 구조를 지원한다. RNNT는 실시간 전사에, CTC는 정밀 전사와 정렬에 사용할 수 있다. NeMo Forced Aligner는 CTC 모델을 이용해 토큰·단어·구간 수준의 타임스탬프를 생성한다. ([NVIDIA Docs](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html?utm_source=chatgpt.com "Models — NVIDIA NeMo Framework User Guide"))

## 실행 전략: 조기 제품 Slice와 핵심 연구를 병행한다

위 프로덕션 아키텍처와 각 전문 모듈의 장기 기술 의존성은 유지한다.
그러나 Gate 1~12를 모두 완료한 뒤 처음 통합하지 않는다.

### Track A: Offline Vertical Slice v0

```text
사용자 WAV
→ 기존 ASR Provider
→ 단순 Surface·방언 표현 평가
→ 구조화된 LLM Tutor 판단
→ 기존 또는 가볍게 적용한 TTS Provider
→ 부산 응답 WAV
```

이 Slice는 최종 모델 품질의 증명이 아니라 전체 학습 경험과 시스템 계약의 검증이다.
초기 ASR, Tutor와 TTS는 사전학습 모델, 외부 API 또는 단순 로직일 수 있으며 안정적인 Provider 뒤에 둔다.

```text
ASRProvider
SimpleAssessmentService
TutorProvider
TTSProvider
OfflineSpeechPipeline
```

### Track B: 핵심 모델 연구

Audio Lab과 고정 Benchmark로 각 후보를 독립 평가하고 부산 특화 모델을 점진적으로 만든다.
`TASK-002 NVIDIA RIVA Korean Conformer-CTC Baseline`, `TASK-003A Surface
ASR Evaluation Calibration`, `TASK-003B Nemotron 3.5 Pretrained Surface ASR
Baseline Integration`과 `TASK-003C Nemotron Inference Configuration
Validation`은 완료됐다. 현재 작업은 `TASK-004 Busan ASR Training Dataset
Foundation`이다. 기존 RIVA와 Nemotron 결과 및 보고서는 삭제하거나 대체하지
않는다. TASK-004 Phase A의 학습 Manifest, split·누출 검증과 모델 중립
export 계약은 구현됐으며, 실제 새 학습 발화 수집은 아직 시작 전이다.

Benchmark 데이터는 평가 전용으로 유지하고 학습 데이터로 사용하지 않는다.
다음 연구 우선순위는 Gate 번호만이 아니라 Offline Vertical Slice와 사용자 테스트에서 확인된 가장 큰 실패를 기준으로 선택한다.

현재 순서는 `TASK-004 Busan ASR Training Dataset Foundation → TASK-005 후보
재판정`이다. TASK-003C에서 빈 출력은 실제 Nemotron blank-only 출력으로
확인됐으므로 Nemotron Fine-tuning 후보는 보류한다. Whisper Cross-model
Comparison은 삭제하지 않고 보류한다.

---

# 3. 각 계층의 정확한 역할

## A. 실시간 음성 인식

### RNNT 또는 TDT Head

사용자가 말하는 동안 대화 엔진이 빠르게 내용을 이해하도록 한다.

```text
사용자: “내 오늘...”
임시 전사: “내 오늘”

사용자: “내 오늘 못 간다.”
최종 실시간 전사: “내 오늘 못 간다”
```

목적은 **빠른 대화 반응**이다.

### CTC Head

발화가 끝난 뒤 실제 표현을 다시 정밀하게 분석한다.

```text
실제 발화
“니 지금 어디고?”

표준어로 자동 수정하지 않고
그대로 보존
```

목적은 다음과 같다.

- 정확한 대화 기록
    
- 부산 표현 인식
    
- 음절·단어 타임스탬프
    
- Forced Alignment
    
- 발음 평가 구간 결정
    
- 모델 오류 분석
    

NeMo는 CTC, RNN-T, TDT, AED 및 하이브리드 디코더를 지원하며, CTC와 Transducer 모델에는 특정 단어나 문구를 강화하는 context biasing도 적용할 수 있다. 부산 핵심 표현 인식을 강화할 때 유용하다. ([NVIDIA Docs](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html?utm_source=chatgpt.com "Automatic Speech Recognition (ASR)"))

---

## B. 발음·억양 분석

Conformer-CTC 하나만으로 발음 점수가 자동으로 완성되는 것은 아니다.

별도의 분석기가 필요하다.

```text
CTC와 Forced Alignment
→ 어느 구간에서 어떤 음절을 말했는지

Phoneme Evaluator
→ 자음·모음·받침을 어떻게 발음했는지

Prosody Evaluator
→ 높낮이·길이·리듬·속도가 어떤지
```

예시 출력:

```json
{
  "surface_text": "니 지금 어디고",
  "target_text": "니 지금 어데고",
  "meaning_passed": true,
  "pronunciation": {
    "consonant": 0.93,
    "vowel": 0.78,
    "final_consonant": 0.88
  },
  "prosody": {
    "intonation": 0.67,
    "rhythm": 0.74,
    "speech_rate": "appropriate"
  },
  "retry_segment": "어데고"
}
```

억양 비교에서는 화자마다 기본 목소리 높이가 다르므로 절대 Hz만 비교하면 안 된다.

```text
잘못된 비교
사용자 160Hz ↔ 원어민 120Hz

필요한 비교
문장 내부의 상대적 높낮이 변화
음절 간 피치 방향
문장 끝 상승·하강 패턴
길이와 쉼의 비율
```

---

## C. Dialect Understanding

이 계층은 단순한 단어 치환기가 아니다.

다음 네 결과를 분리해야 한다.

```text
실제 발화
“니 지금 어디고?”

표준어 의미
“너 지금 어디야?”

권장 부산 표현
“니 지금 어데고?”

상황 판정
친한 친구에게 위치를 묻는 표현
```

여기서 평가하는 것은:

- 부산 어휘 사용
    
- 종결형
    
- 높임말과 반말
    
- 상대와의 관계
    
- 상황 적절성
    
- 사투리 강도
    
- 자연스러움
    

이다.

같은 문장도 상대방에 따라 판정이 달라진다.

```text
친구에게
“니 밥 묵었나?”
→ 자연스러움

처음 만난 어른에게
“니 밥 묵었나?”
→ 문법은 가능해도 관계상 부적절
```

---

# 4. 숨은 핵심: Tutor Coordinator

이것이 네 서비스와 일반 AI 음성 채팅을 갈라놓는 가장 중요한 계층이다.

Tutor Coordinator 안에는 네 가지 시스템이 들어간다.

## Scenario Engine

Step-up 콘텐츠를 상태 머신으로 관리한다.

```text
상황 소개
→ 따라 말하기
→ 유도 말하기
→ 자유 응답
→ 상황 미션
→ 피드백
→ 다음 에피소드
```

각 단계에는 명확한 조건이 있다.

```json
{
  "mission": "친구에게 10분 늦는다고 설명하기",
  "required_intents": [
    "delay_explanation",
    "arrival_time",
    "request_to_wait"
  ],
  "recommended_dialect_terms": [
    "쪼매",
    "기다려라"
  ],
  "relationship": "friend"
}
```

문장 하나와 정확히 일치해야 통과하는 것이 아니라, 의도와 상황 목표를 만족하면 통과한다.

---

## Learner Model

사용자의 장기 학습 상태를 저장한다.

```json
{
  "expressions": {
    "어데": {
      "comprehension": 0.9,
      "production": 0.52,
      "error_count": 4
    },
    "쪼매": {
      "comprehension": 0.85,
      "production": 0.72,
      "error_count": 1
    }
  },
  "prosody": {
    "sentence_final_intonation": 0.48,
    "rhythm": 0.66
  },
  "contexts": {
    "friend": 0.82,
    "restaurant": 0.63,
    "elder": 0.31
  }
}
```

이 값은 정확한 시험 성적이라기보다 다음 수업을 결정하기 위한 내부 상태다.

---

## Teaching Policy

오류를 찾는 것과 가르치는 것은 다르다.

Teaching Policy는 다음 중 하나를 선택한다.

```text
대화를 그대로 진행
올바른 표현을 캐릭터가 자연스럽게 재사용
노란색 Tip 표시
한 문장만 다시 말하게 함
대화 종료 후 복습에 저장
다음 에피소드에서 같은 표현 재등장
난이도 또는 사투리 강도 조정
```

예시 정책:

```text
의미 전달 성공 + 첫 번째 작은 오류
→ 대화를 중단하지 않음
→ natural recast + Tip

같은 오류 세 번 반복
→ 짧은 직접 교정 + 재시도

의미 전달 실패
→ 캐릭터가 되묻기
→ 힌트 제공

관계상 부적절한 표현
→ 대화는 이어가되 관계 Tip 표시

심한 발음 오류
→ 해당 구간만 다시 말하게 함
```

초기에는 규칙 기반으로 운영하고, 교정 데이터가 쌓이면 분류 또는 랭킹 모델로 발전시킨다.

---

## Dialect Knowledge Base

모델의 기억에만 부산 사투리 정답을 맡기면 일관성이 떨어진다.

표현마다 구조화된 기준이 있어야 한다.

```json
{
  "expression": "어데고?",
  "standard_meaning": "어디야?",
  "region": ["Busan", "Gyeongsang"],
  "relationship": ["friend", "close_peer"],
  "formality": "informal",
  "dialect_strength": 2,
  "variants": [
    "어데 있노?",
    "지금 어데고?"
  ],
  "avoid_contexts": [
    "first_meeting_with_elder"
  ],
  "reference_audio": [
    "speaker_021_utt_104.wav"
  ]
}
```

이 지식베이스는:

- AI 응답 생성
    
- 사용자 발화 평가
    
- Tip 생성
    
- 복습 콘텐츠 생성
    
- 데이터 라벨링
    

의 공통 기준이 된다.

---

# 5. 부산 사투리 음성 생성 모델

사용자가 최종적으로 듣는 음성이 가장 중요하다.

일반 TTS처럼 부산 문장을 읽기만 하는 수준을 넘어서야 한다.

```text
입력
부산 문장
캐릭터 ID
상대 관계
사투리 강도
감정
말하기 속도
운율 계획

출력
실제 부산 화자처럼 들리는 음성
```

## 초기 생성 구조

```text
Dialect Dialogue Planner
→ 부산 사투리 텍스트
→ G2P와 운율 계획
→ Dialect TTS
→ Vocoder 또는 audio codec decoder
→ 음성
```

NeMo는 TTS 모델 학습·파인튜닝과 데이터 전처리를 지원하고, TTS 모델 및 음성 코덱 관련 구성도 제공한다. NeMo에서 개발한 음성 모델은 Riva 또는 NVIDIA Speech NIM 계열로 프로덕션 배포하는 경로를 사용할 수 있다. ([NVIDIA Docs](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tts/intro.html?utm_source=chatgpt.com "Text-to-Speech (TTS) — NVIDIA NeMo Framework User ..."))

## 장기 생성 구조

```text
Speech Encoder
→ Speech-Language Core
→ Speech Token Decoder
→ 부산 사투리 음성
```

NVIDIA NeMo SpeechLM2 문서에도 오디오 인지, 언어 모델, 음성 합성을 결합한 end-to-end S2S 구조가 포함되어 있다. 네 최종 연구 방향은 이 구조를 부산 사투리와 교육 목적에 특화하는 것이다. ([NVIDIA Docs](https://docs.nvidia.com/nemo/speech/nightly/speechlm2/models.html?utm_source=chatgpt.com "Models — NeMo-Speech"))

다만 프로덕션 첫 버전에서는 내부 텍스트와 교육 판정을 관찰할 수 있는 **하이브리드 Cascade 구조**를 권장한다.

```text
겉으로는 S2S처럼 빠르게 작동
내부적으로는 전사·평가·응답 계획을 관찰 가능
```

교육 서비스는 잘못된 교정을 추적하고 수정할 수 있어야 하기 때문이다.

---

# 6. OpenAI Realtime의 역할

최종 목표는 자체 사투리 엔진이지만, 개발 과정에서는 OpenAI Realtime을 다음 용도로 사용한다.

```text
초기 대화 추론 엔진
캐릭터 행동 구현
Teaching Policy 도구 호출
자체 모델 비교용 Teacher
자체 엔진 실패 시 fallback
합성 대화 데이터 초안 생성
```

OpenAI Realtime API는 WebRTC, WebSocket, SIP 연결을 지원하며, 브라우저·모바일 음성 애플리케이션에는 WebRTC가 권장된다. Realtime 세션은 오디오를 송수신하고 대화 상태를 유지하며 외부 도구를 호출할 수 있다. ([OpenAI Developers](https://developers.openai.com/api/docs/guides/realtime-webrtc?utm_source=chatgpt.com "Realtime API with WebRTC"))

등록할 도구의 예시는 다음과 같다.

```text
get_current_mission()
get_learner_profile()
evaluate_utterance(audio_id)
select_teaching_action(error_report)
save_learning_result()
get_dialect_reference(expression_id)
```

Function calling을 사용하면 모델이 외부 평가 시스템과 학습 상태 DB를 호출한 뒤 그 결과를 대화에 반영할 수 있다. ([OpenAI Developers](https://developers.openai.com/api/docs/guides/function-calling?utm_source=chatgpt.com "Function calling | OpenAI API"))

OpenAI Realtime의 음성 출력을 최종 부산 음성으로 사용하지 않고, 응답 계획만 받아 자체 Dialect TTS로 출력하는 구성도 가능하다.

```text
OpenAI Realtime 또는 LLM
→ 부산 응답 계획과 텍스트
→ 자체 부산 TTS
→ 최종 음성
```

---

# 7. 필요한 데이터셋 4종

모델 하나를 위한 데이터가 아니라, 네 개의 서로 다른 데이터 자산을 만들어야 한다.

## ① ASR 데이터

목적은 사용자가 실제로 말한 표현을 보존하는 것이다.

```text
음성
실제 전사
화자 ID
지역
녹음 환경
말하기 속도
사투리 강도
```

## ② TTS 데이터

목적은 자연스러운 부산 음성을 생성하는 것이다.

```text
깨끗한 스튜디오 음성
정확한 사투리 문장
화자 정보
감정
관계
사투리 강도
피치·길이·쉼
```

ASR 데이터는 소음과 다양한 화자가 중요하지만, TTS 데이터는 깨끗하고 일관된 녹음 품질이 중요하다.

## ③ 사투리 대화 데이터

목적은 상황에 맞는 표현을 생성하는 것이다.

```text
이전 대화
상황
화자 관계
표준어 의미
가능한 부산 응답
부적절한 응답
사투리 강도
```

## ④ 교육 개입 데이터

목적은 어떻게 교정할지 학습하는 것이다.

```text
사용자 상태
사용자 발화
오류 종류
대화 상황
가능한 교정 방식
가장 좋은 교정
부산 원어민 또는 교육 전문가 평가
```

예:

```json
{
  "learner_utterance": "니 지금 어디고?",
  "target": "니 지금 어데고?",
  "error_type": "dialect_lexical_substitution",
  "severity": "low",
  "recommended_action": "natural_recast",
  "character_response": "내 서면이다. 니는 어데고?",
  "ui_tip": "부산에서는 ‘어데고?’가 더 자연스러워요."
}
```

---

# 8. 부산 원어민 검수 체계

부산 화자 한 명의 말투를 전체 부산 사투리의 정답으로 사용하면 안 된다.

최소한 다음 구성을 갖춘 패널을 만든다.

```text
20대 부산 화자
30~40대 부산 화자
50대 이상 부산 화자
성별과 생활권이 다른 화자
언어교육 또는 국어·음성학 전문가
```

이들은 다음을 평가한다.

```text
표현이 실제로 자연스러운가?
관계와 상황에 적절한가?
사투리 강도가 맞는가?
음성 억양이 부산 사람처럼 들리는가?
교정 방식이 부담스럽지 않은가?
```

평가자 의견이 갈리는 표현은 정답 하나로 고정하지 않는다.

```text
허용 표현 여러 개
세대별 변형
관계별 변형
사투리 강도별 변형
```

으로 관리한다.

---

# 9. 반드시 먼저 만들 Benchmark

모델 학습 전에 시험지를 만든다.

## ASR Benchmark

```text
CER
핵심 부산 표현 Recall
사투리 보존 정확도
사투리→표준어 오보정률
화자 그룹별 성능
소음 환경별 성능
p50·p95 지연시간
```

## TTS Benchmark

```text
자연스러움 MOS
부산 사투리 자연스러움
억양 적합성
캐릭터 음색 일관성
상황·감정 적합성
긴 대화에서의 안정성
```

## Teaching Benchmark

```text
오류 탐지 정확도
교정 내용 정확도
개입 타이밍 적절성
부산 원어민·교사 평가 일치도
불필요한 교정 비율
재연습 후 개선율
```

## 전체 시스템 Benchmark

```text
사용자 발화 종료
→ AI 첫 음성까지의 지연

ASR 오류 발생 시 복구율
30분 연속 대화 안정성
끼어들기 처리
VAD 조기 종료율
도구 호출 실패율
세션 복원
```

NeMo는 ASR 모델 비교기, ASR Evaluator, Speech Data Explorer, Forced Aligner 등 음성 모델 개발에 필요한 도구를 제공한다. ([NVIDIA Docs](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tools/intro.html?utm_source=chatgpt.com "Speech AI Tools — NVIDIA NeMo Framework User Guide"))

---

# 10. 가장 빠른 병렬 실행 구조

네가 순서대로 모든 일을 처리하면 느리다. 다섯 개 트랙을 동시에 돌린다.

## 연구 Workstream 1: 언어·데이터

```text
부산 핵심 표현 300개
상황 30개
관계 유형
사투리 강도
라벨링 가이드
원어민 패널
```

## 연구 Workstream 2: ASR

```text
Baseline 비교
FastConformer Hybrid CTC/RNNT
Context biasing
Tokenizer 실험
파인튜닝
Forced Alignment
```

## 연구 Workstream 3: 음성 생성

```text
부산 화자 데이터 수집
G2P와 운율 표현
Dialect TTS
Vocoder 또는 codec decoder
자연스러움 평가
```

## 연구 Workstream 4: 교육 지능

```text
Scenario Engine
Learner Model
오류 taxonomy
Teaching Policy
교정 데이터
복습 알고리즘
```

## 연구 Workstream 5: 프로덕션

```text
WebRTC
실시간 오케스트레이터
GPU 서빙
DB
로그
모델 버전 관리
보안·동의
부하 테스트
```

---

# 11. AI 에이전트 운영 방식

너는 코드를 전부 직접 치는 사람이 아니라 **연구 책임자와 시스템 설계자**가 된다.

```text
Architect Agent
전체 인터페이스와 데이터 계약 관리

Data Agent
manifest·전처리·분할·검증

ASR Agent
NeMo 학습과 디코딩 실험

Speech Agent
TTS·운율·코덱 실험

Evaluation Agent
지표·오류 분석·리포트

Production Agent
API·GPU 서빙·WebRTC·Docker

Skeptic Agent
데이터 누수·과적합·결과 착시 공격
```

각 실험에는 반드시 이 문서가 붙는다.

```text
실험 ID
가설
변경 변수 한 개
고정 조건
데이터셋 버전
모델 체크포인트
평가 지표
결과
실패 사례
다음 행동
```

절대 규칙:

```text
AI가 테스트 세트를 수정하지 못하게 함
train·validation·test 화자 중복 금지
실험당 주요 변수 하나만 변경
실패 실험도 기록
모델과 데이터 버전을 함께 저장
다른 Agent가 결과를 검증
```

---

# 12. 공격적인 12주 실행 계획

기간은 성공 여부를 보장하는 약속이 아니라, 병렬 실행을 위한 전투 계획이다.

## 1주차: 사양 동결

산출물:

```text
시스템 아키텍처 v1
데이터 스키마
부산 표현 300개
상황 30개
오류 taxonomy
Benchmark v0.1
출시 통과 기준
```

## 2주차: Baseline 비교

```text
여러 사전학습 ASR 평가
여러 TTS 기준 모델 평가
사투리 오보정 사례 수집
최종 출발 체크포인트 선택
```

NeMo는 공개된 사전학습 체크포인트를 사용하거나 자체 ASR을 학습할 수 있고, 설정 파일을 통해 다양한 모델 구조와 파인튜닝 실험을 구성할 수 있다. ([NVIDIA Docs](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/configs.html?utm_source=chatgpt.com "NeMo ASR Configuration Files"))

## 3~4주차: 자체 ASR v1

```text
FastConformer Hybrid CTC/RNNT
부산 표현 context biasing
사투리 보존 전사
단어 타임스탬프
CER와 사투리 Recall 측정
```

## 3~5주차: 부산 TTS v1 병렬 개발

```text
화자 데이터 정리
음성 품질 검증
사투리 문장·운율 라벨
사전학습 TTS 파인튜닝
부산 자연스러움 평가
```

## 5~6주차: 음소·억양 분석

```text
Forced Alignment
자모·음절 구간
F0·길이·에너지 추출
발음 오류 v1
억양 비교 v1
```

## 6~7주차: Dialect Understanding

```text
표면 전사
표준어 의미
부산 목표 표현
관계 적절성
사투리 강도
자연스러움
```

을 하나의 구조화된 출력으로 생성한다.

## 7~8주차: Teaching Engine

```text
오류 심각도
교정 시점
natural recast
Tip
재시도
복습 저장
```

규칙 기반 정책을 먼저 완성하고 원어민 패널이 검수한다.

## 8~9주차: Dialogue Core 통합

```text
현재 상황
학습 목표
사용자 상태
발화 분석 결과
→ 부산 캐릭터 응답 계획
```

초기에는 OpenAI Realtime 또는 텍스트 LLM을 이용하되 출력은 자체 Dialect TTS로 통제한다.

## 9~10주차: 실시간 통합

```text
WebRTC
VAD
ASR streaming
CTC 후처리
Teaching tool calls
자체 TTS streaming
끼어들기
오디오 취소
```

## 11주차: 실패 사례 집중 학습

잘된 샘플보다 다음 오류만 공격한다.

```text
어데 → 어디
묵었나 → 먹었나
주이소 → 주세요
긴 침묵
빠른 발화
타지역 한국인
외국인 한국어
카페·거리 소음
관계상 부적절한 표현
부자연스러운 부산 TTS 억양
```

## 12주차: RC 동결 및 검증

```text
Dataset v1.0
ASR RC1
TTS RC1
Teaching Policy RC1
Inference API RC1
Benchmark v1.0
전체 재현 보고서
```

새로운 화자, 다른 스마트폰, 소음 환경, 긴 세션, 동시 요청으로 검증한다.

---

# 13. 앱 개발에 들어가는 기술 게이트

네 전략대로라면 다음 기준을 통과한 뒤 본격적인 앱 화면 개발에 들어간다.

```text
Gate 1
고정된 부산 Benchmark 존재

Gate 2
자체 ASR이 선택한 범용 baseline보다 우수

Gate 3
사투리 표현을 표준어로 잘못 고치는 비율이 기준 이하

Gate 4
부산 TTS 자연스러움을 원어민 패널이 승인

Gate 5
발음·억양 평가가 인간 평가와 충분히 일치

Gate 6
Teaching Policy의 불필요한 개입이 기준 이하

Gate 7
실시간 지연 기준 통과

Gate 8
30분 이상 안정적 대화

Gate 9
데이터·모델·실험이 완전히 재현 가능

Gate 10
음성·카메라 개인정보 처리 체계 완성
```

---

# 14. 서비스 운영에 필요한 모델 밖의 시스템

모델만 좋아서는 서비스를 운영할 수 없다.

## 모델 운영

```text
Dataset Registry
Model Registry
실험 추적
배포·Rollback
A/B 테스트
GPU Autoscaling
Fallback 모델
```

## 관측

```text
ASR 정확도
사투리 Recall
TTS 생성 실패
p50·p95 지연
VAD 조기 종료
사용자 끼어들기
Tip 표시 횟수
재시도율
Step 통과율
```

## 개인정보

```text
음성 저장 동의
모델 학습 활용 동의 별도
카메라 영상 저장 여부
데이터 삭제 요청
보관 기간
암호화
사용자별 데이터 분리
```

## 실패 복구

```text
자체 ASR 실패
→ 대체 전사 모델

자체 TTS 실패
→ 안전한 기본 음성

Teaching Engine 불확실
→ 교정하지 않고 대화만 진행

모델 서버 지연
→ 짧은 캐릭터 반응 또는 재연결 안내
```

---

# 15. 최종 저장소 구조

```text
busan-ai/
├── apps/
│   ├── mobile/
│   └── admin-console/
│
├── services/
│   ├── realtime-orchestrator/
│   ├── learner-service/
│   ├── scenario-service/
│   └── analytics-service/
│
├── models/
│   ├── asr/
│   ├── phoneme-evaluator/
│   ├── prosody-evaluator/
│   ├── dialect-understanding/
│   ├── teaching-policy/
│   └── dialect-tts/
│
├── data/
│   ├── schemas/
│   ├── manifests/
│   ├── labeling-guides/
│   └── benchmarks/
│
├── knowledge/
│   ├── expressions/
│   ├── scenarios/
│   ├── relationships/
│   └── pronunciation/
│
├── evaluation/
│   ├── asr/
│   ├── tts/
│   ├── teaching/
│   └── end-to-end/
│
├── infra/
│   ├── training/
│   ├── inference/
│   ├── monitoring/
│   └── deployment/
│
└── experiments/
    ├── registry/
    └── reports/
```

---

# 최종 결론

네가 만들어야 하는 최종 시스템은 다음 공식으로 정리할 수 있다.

```text
부산 네이티브 음성 AI
+
자체 Hybrid ASR
+
발음·억양 평가기
+
사투리 지식베이스
+
Teaching Intelligence
+
Learner Model
+
Step-up Scenario Engine
+
실시간 프로덕션 플랫폼
```

그리고 아래 순서는 장기 기술 의존성과 최종 성숙도 순서로 유지한다.
제품 개발은 이 목록을 모두 완료한 뒤 처음 통합하는 방식이 아니다.

```text
1. 정답 기준과 Benchmark
2. 듣는 귀인 ASR
3. 말하는 입인 부산 TTS
4. 발음·억양 분석
5. 사투리 이해
6. Teaching Engine
7. Learner Model과 Step-up
8. 실시간 통합
9. 실패 공격과 RC 검증
10. 앱 인터페이스 개발
```

현재 실행 순서는 다음과 같다.

```text
1. TASK-002 RIVA Korean Conformer-CTC Baseline 완료 및 보존
2. TASK-003A Surface ASR Evaluation Calibration 완료 및 보존
3. TASK-003B Nemotron 3.5 pretrained Baseline 완료 및 보존
4. TASK-003C Nemotron 빈 출력 원인 검증 완료 및 보존
5. TASK-004 Busan ASR Training Dataset Foundation 진행(Phase A 완료)
6. 데이터 기반 준비 후 TASK-005 Fine-tuning 후보 재판정
7. Track A Offline Vertical Slice와 사용자 흐름에서 가장 큰 실패 측정
8. Provider 내부 구현을 부산 특화 모델로 점진 교체
```

정교한 IPA, Forced Alignment, Prosody, 자체 부산 TTS와 WebRTC는 최종 아키텍처에 남아 있지만 `Offline Vertical Slice v0`의 선행 조건은 아니다.

가장 중요한 원칙은 하나다.

> **AI가 부산 사람처럼 말하는 것과 사용자를 정확하게 가르치는 것을 하나의 모델에게 막연히 맡기지 않는다. 각각을 측정 가능한 전문 계층으로 만들고, Tutor Coordinator가 하나의 사람처럼 조율하게 한다.**

화면에는 부산 친구 한 명만 보인다. 그러나 뒤에서는 정밀한 귀, 부산 언어학자, 발음 코치, 수업 설계자, 장기 기억이 한 팀으로 움직인다. 그것이 네 사투리 AI의 최종 설계다.
