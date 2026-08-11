# Busan Speech Research Lab v0

`TASK-001`의 산출물이다. 부산 사투리 Surface ASR을 학습시키기 전에 원본
음성, 언어 정답, 모델 가설과 실패 유형을 같은 계약으로 보존하고 측정한다.

현재 연구 질문은 다음 하나다.

> 고정된 Nemotron pretrained와 TASK-005 Busan encoder adapter의 차이가
> 재현 가능하며, 보지 않은 다화자 부산어에 일반화하고 표준 한국어를 훼손하지 않는가?

`busan-surface-v0@1.0.0` 10개와 외부 NVIDIA GPU 환경의 prediction을 분리해
주고받는 측정 수직 슬라이스를 제공한다. 모델 런타임과 Audio Lab은 같은
컴퓨터에 있을 필요가 없다.

## Gate 2 상태 (2026-08-09)

**FAIL (검증 증거 미완료)**. TASK-005 학습과 10개 자동 평가는 완료됐지만 Gate 2는
아직 종료하지 않는다.

- Benchmark 두 원본은 raw SHA가 다르지만 검증된 의미 내용은 동일하다. Canonical
  package SHA-256은 `151c1e28804627bea69bbd7f6632f4d3558ebf076147e42c1d168d508467233c`,
  semantic SHA-256은 `700d352edb4a4e9321b48ec6cd312bec6ad1d4c48fa2bedbcf80a2ca23a67f8c`다.
- 기존 Prediction을 재추론 없이 재평가한 결과는 pretrained CER `0.6167`, 방언 보존
  `0.2000`, 빈 출력 3개; adapter CER `0.0500`, 방언 보존 `0.9333`, 빈 출력 0개다.
- GPU recovery bundle의 125/125 파일과 best `.ckpt`, best `.nemo`, embedded adapter,
  hparams, launcher, inference runner, 환경 및 학습 로그 SHA-256 검증이 통과했다.
- 10개 blinded Human A/B는 준비됐지만 판정은 0/10이다.
- checkpoint 선택에 쓴 Validation 40개는 final Test에서 제외한다. 독립 다화자 부산어
  Test와 표준 한국어 Regression 데이터·결과는 아직 없다.

감사 결과, 복구 목록, 검수 파일과 Gate 기준은
[`artifacts/gate-2/`](artifacts/gate-2/README.md)에 있다. Gate 2가 PASS하기 전에는
Streaming ASR, IPA, Forced Alignment, 발음 점수 또는 TTS로 넘어가지 않는다.

## 현재 아키텍처 위치

```text
업로드 원본
  → 검증 / 원본 바이트 보존 / SHA-256
  → 48kHz mono 또는 stereo PCM24 마스터
      ├─ ASR 학습용 16kHz mono PCM16 WAV
      ├─ 발음 평가용 24kHz mono PCM16 WAV
      └─ TTS 학습용 48kHz mono PCM24 WAV
  → 공통 Utterance Schema
  → 고정 Benchmark Manifest
  → Surface ASR Adapter
  → CER / 부산 표현 보존 / 과보정
  → 고신뢰 오답과 실패 샘플 export
```

최종 아키텍처의 `Data Platform + Evaluation Platform + Surface-form ASR`
경계에 해당한다. Streaming ASR, Direct IPA, 신경 Prosody, TTS 학습은 포함하지
않는다.

## 구현 범위

- 원본 오디오를 콘텐츠 주소형 경로에 그대로 보존
- FFprobe 디코딩·길이·채널·샘플레이트 검증
- FFmpeg 48kHz PCM24 마스터와 ASR·발음·TTS 목적별 WAV 생성
- 원본→마스터→파생본의 SHA-256, 부모 hash, 변환 fingerprint와 품질 지표
- Pydantic `UtteranceRecord`, `BenchmarkManifest`, ASR/평가 계약
- `surface_text`와 `normalized_meaning`의 독립 필드
- 저장/연구/학습 동의를 분리한 Consent Schema
- 화자와 오디오 계보가 split을 넘지 못하게 하는 manifest validator
- 교체 가능한 `SurfaceASRAdapter`
- 사전 계산 prediction JSONL용 adapter
- CER, Dialect Preservation Rate, Context Overcorrection Rate
- high-confidence wrong case와 normalization error JSONL export
- Waveform, log-Mel, 탐색적 F0를 보여주는 로컬 연구 UI
- 단위·통합·회귀 테스트와 JSON Schema export

## 설치

요구 사항:

- Python 3.13.14 (`.python-version`으로 고정)
- [`uv`](https://docs.astral.sh/uv/)
- FFmpeg와 FFprobe가 `PATH`에 존재

프로젝트 전용 Skill로 설치·검사·실행을 한 명령으로 통일한다.

```bash
# 최초 실행 또는 소스/의존성 변경 뒤
python3 ~/.codex/skills/use-busan-project-venv/scripts/run.py --sync busan-lab doctor

# 이후 일반 실행
python3 ~/.codex/skills/use-busan-project-venv/scripts/run.py busan-lab serve --port 8000
```

실제 패키지는 `~/.cache/busan-speech-research-lab/.venv`에만 설치되고 프로젝트의
`.venv`는 그 위치를 가리키는 링크다. 따라서 다른 프로젝트 패키지와 섞이지 않으며,
Desktop 동기화 폴더 안에 수천 개의 패키지 파일을 만들지 않는다. wrapper는
`.python-version`, `uv.lock`, FFmpeg/FFprobe, 활성 환경을 실행 전 검사한다.
전역 `python`이나 `pip`로 이 프로젝트 패키지를 설치하지 않는다.

의존성 선택 이유:

- FastAPI: 업로드/API/정적 연구 UI를 하나의 작은 프로세스로 유지
- Pydantic: 공통 객체를 중복 없이 버전된 계약으로 검증
- NumPy: 모델 의존성 없이 Wave/Mel/F0와 평가 집계를 재현
- python-multipart: 브라우저 원본 업로드
- pytest/httpx: API를 포함한 오프라인 수직 슬라이스 검증
- ruff/mypy: 스타일과 타입 계약 검증
- Hatchling: non-editable wheel과 정적 UI package data를 재현

NeMo, PyTorch, librosa는 `TASK-001`의 병목이 아니므로 추가하지 않았다.

## 실행

```bash
python3 ~/.codex/skills/use-busan-project-venv/scripts/run.py \
  busan-lab serve --host 127.0.0.1 --port 8000
```

브라우저에서 <http://127.0.0.1:8000>을 연다. OpenAPI 계약은
<http://127.0.0.1:8000/docs>에서 확인할 수 있다.

기본 데이터 경로는 `data/lab`이다. 다른 위치를 사용하려면:

```bash
BUSAN_LAB_DATA_DIR=/absolute/path/to/lab-data \
  python3 ~/.codex/skills/use-busan-project-venv/scripts/run.py busan-lab serve
```

## 입력과 출력

### Utterance 입력

- 사용자 원본 오디오
- 화자 ID, 지역, 기기, 환경
- 실제 발화 형태인 `surface_text`
- 의미를 별도로 적는 `normalized_meaning`
- 부산 표현과 표준화 후보
- 저장/연구/모델 학습 동의

### 저장 구조

```text
data/lab/
├── raw/<sha-prefix>/<original-sha>.<ext>
├── master/<sha-prefix>/<original-sha>.master-48k-{mono|stereo}.wav
├── derived/
│   ├── asr_16k_mono/<sha-prefix>/<original-sha>.asr-16k-mono.wav
│   ├── pronunciation_24k_mono/<sha-prefix>/<original-sha>.pronunciation-24k-mono.wav
│   └── tts_48k_mono/<sha-prefix>/<original-sha>.tts-48k-mono.wav
├── records/<utterance-id>.json
├── label_revisions/<utterance-id>/<revision-id>.json
├── manifests/<benchmark-id>--<version>.json
├── experiments/<experiment-id>.json
├── predictions/<prediction-id>.json
├── reviews/<review-id>.json
├── calibrations/<evaluation-revision>.json
├── trash/<utterance-id>/<archive-id>/...
├── reports/<report-id>.json
└── exports/normalization-errors.jsonl
```

업로드 원본과 label version은 덮어쓰지 않는다. 같은 원본 바이트는 SHA-256으로
deduplicate하며, 마스터는 업로드 원본을, 목적별 WAV는 마스터를 부모로 기록한다.
`audio_contract_version=1.1.0`은 기존 `original + derived_16k_mono` 레코드도
읽을 수 있는 additive migration이다. 기존 파일을 자동으로 덮어쓰지는 않는다.

### 핵심 언어 계약

```json
{
  "surface_text": "국밥 하나 주이소",
  "normalized_meaning": "국밥 하나 주세요",
  "dialect_expressions": [
    {
      "surface_form": "주이소",
      "normalized_forms": ["주세요"],
      "status": "candidate"
    }
  ]
}
```

LLM이나 규칙이 만든 부산 표현은 기본적으로 `candidate`다. 원어민/전문가
검수 후에만 `approved`로 올린다. 평가 결과는 전체 후보 수와 승인 수를 따로
기록한다.

## Benchmark 고정

연구 UI에서 현재 발화를 test manifest로 고정하거나 API를 호출한다.

```bash
curl -X POST http://127.0.0.1:8000/api/benchmarks \
  -H 'content-type: application/json' \
  -d '{
    "benchmark_id": "busan-surface-v0",
    "benchmark_version": "0.1.0",
    "utterance_ids": ["PUT-UTTERANCE-UUID-HERE"],
    "split": "test"
  }'
```

Manifest 생성 시 다음을 거부한다.

- 동일 `utterance_id` 중복
- 같은 `speaker_id`의 train/validation/test 교차
- 같은 원본 또는 파생 음성 hash의 split 교차
- `unassigned` 상태의 benchmark entry
- 이미 존재하는 frozen benchmark version의 내용 변경

## Surface ASR prediction 계약

실제 모델 실행은 이 저장소 바깥의 GPU 환경에서 수행할 수 있다. 결과를 다음
JSONL로 export하면 측정 엔진은 모델 런타임과 분리된 상태에서 평가한다.

```json
{"experiment_id":"task-002-nvidia-korean-conformer-ctc-pretrained-v0","benchmark_id":"busan-surface-v0","benchmark_version":"1.0.0","utterance_id":"00000000-0000-0000-0000-000000000000","audio_sha256":"64-char-derived-sha","device":"NVIDIA GPU and runtime description","inference_timestamp":"2026-07-29T00:00:00Z","result":{"schema_version":"1.0.0","surface_text":"국밥 하나 주세요","confidence":null,"confidence_supported":false,"latency_ms":142.0,"model":{"name":"RIVA Conformer ASR Korean","version":"deployable_v1.0","model_provider":"NVIDIA","model_family":"Conformer-CTC","decoder_type":"CTC greedy","fine_tuned":false,"checkpoint_identifier":"nvidia/tao/speechtotext_ko_kr_conformer:deployable_v1.0","checkpoint":null,"tokenizer_version":null,"config_hash":null},"segments":[]}}
```

고정 Benchmark와 hash 검증된 파생 WAV를 외부 GPU 실행용 ZIP으로 만든다.

```bash
python3 ~/.codex/skills/use-busan-project-venv/scripts/run.py busan-lab export-benchmark \
  --benchmark-id busan-surface-v0 \
  --benchmark-version 1.0.0 \
  --output artifacts/task-002/busan-surface-v0--1.0.0.zip
```

검증 Notebook은
`notebooks/task_002_nvidia_korean_conformer_ctc.ipynb`다. 현재 공식 NGC
공개 페이지에서 확인된 후보와 아직 확인되지 않은 파일·런타임 조건을 분리해
기록하고, 외부 Riva 실행 결과가 Import 계약과 일치하는지 검사한다.
confidence를 신뢰할 수 있게 산출하지 않는 런타임은 `confidence: null`,
`confidence_supported: false`로 기록한다.

공식 후보 상태:

```text
model_name: RIVA Conformer ASR Korean
checkpoint_identifier: nvidia/tao/speechtotext_ko_kr_conformer:deployable_v1.0
format: Riva deployment model; exact downloadable filename pending verification
Python inference: Riva client는 가능하지만 Riva server/runtime가 별도로 필요
access: NGC sign-in required
license: NVIDIA Riva license
```

NGC 파일 브라우저 로그인 뒤 정확한 파일명·hash와 호환 Riva 버전을 검증하기
전에는 `.nemo` 또는 NeMo `restore_from` 경로를 추측해 넣지 않는다.

```bash
python3 ~/.codex/skills/use-busan-project-venv/scripts/run.py busan-lab evaluate \
  --benchmark-id busan-surface-v0 \
  --benchmark-version 1.0.0 \
  --predictions /absolute/path/predictions.jsonl
```

한 prediction 파일에는 정확히 한 experiment ID와 모델 버전만 있어야 한다.
Import는 prediction의 `(utterance_id, audio_sha256)` 집합이 고정 Benchmark와
정확히 일치하지 않으면 거부한다.

## TASK-004 학습 데이터 계약

기존 `busan-surface-v0@1.0.0`은 test-only Benchmark이며 학습 Manifest에
들어갈 수 없다. TASK-004는 새 UI를 만들지 않고 기존 업로드의 별도
`model_training_allowed` 동의, 오디오 lineage, Surface label과 음질 정보를
재사용한다. Phase A의 계약·검증·내보내기 구현은 완료됐으며, 실제 새
학습 발화 300~500개 수집은 아직 시작 전이다.

학습 가능 조건:

```text
명시적 model_training_allowed 동의
Surface label_status가 human_reviewed 또는 approved
Audio Lab 품질 검사 통과
train/validation 명시
frozen Benchmark와 utterance/speaker/audio lineage/동일 문장 비중복
train/validation 사이 speaker/audio lineage/동일 문장 비중복
```

새 발화를 원음과 대조한 뒤 UI를 추가하지 않고 CLI로 검수 상태를 확정한다.

Voice Memos에서 순서대로 저장된 단일 화자 M4A 묶음은 먼저 dry-run으로
검사한다. `--commit`이 없으면 원본 변환이나 Audio Lab 등록은 일어나지 않는다.

```bash
python3 ~/.codex/skills/use-busan-project-venv/scripts/run.py \
  busan-lab import-training-recordings \
  --import-id task-004-solo-speaker-001-v0 \
  --input-dir /absolute/path/busan_Audio \
  --prompt-sheet artifacts/task-004/SOLO_SPEAKER_300.md \
  --prompt-start 1 \
  --prompt-end 200 \
  --speaker-id busan-train-speaker-001 \
  --region Busan \
  --device "Apple Voice Memos" \
  --recording-environment quiet_room \
  --confirm-storage-consent \
  --confirm-research-use \
  --confirm-model-training-consent
```

검사가 통과한 같은 명령에 `--commit`을 추가하면 M4A 원본을 보존하고 목적별
WAV를 생성한 뒤, `candidate` Surface label과 append-only Import ledger를
저장한다. macOS에서 파일이 `dataless`라면 Finder의 `지금 다운로드`를 먼저
실행한다. 한 화자의 묶음은 전부 train으로만 등록된다.

```bash
python3 ~/.codex/skills/use-busan-project-venv/scripts/run.py \
  busan-lab review-training-label \
  --utterance-id <새-학습-발화-ID> \
  --reviewer-id <검수자-ID> \
  --status approved \
  --reason "원음과 Surface transcript 대조 완료"
```

`artifacts/task-004/training-split-assignments.example.json`을 복사해 실제 새
발화 ID로 교체하고 화자 단위로 train/validation을 나눈다.

```bash
python3 ~/.codex/skills/use-busan-project-venv/scripts/run.py \
  busan-lab create-training-dataset \
  --dataset-id busan-asr-training-pilot-v0 \
  --dataset-version 0.1.0 \
  --assignments /absolute/path/training-split-assignments.json
```

생성된 Manifest는 immutable하며 다른 내용으로 같은 version을 덮어쓸 수 없다.
다시 검증하거나 모델 중립 ZIP을 만들 수 있다.

```bash
python3 ~/.codex/skills/use-busan-project-venv/scripts/run.py \
  busan-lab validate-training-dataset \
  --dataset-id busan-asr-training-pilot-v0 \
  --dataset-version 0.1.0

python3 ~/.codex/skills/use-busan-project-venv/scripts/run.py \
  busan-lab export-training-dataset \
  --dataset-id busan-asr-training-pilot-v0 \
  --dataset-version 0.1.0 \
  --output /absolute/path/busan-asr-training-pilot-v0--0.1.0.zip
```

Export ZIP은 다음을 포함한다.

```text
training_dataset.json
validation_report.json
schemas/
manifests/train.jsonl
manifests/validation.jsonl
derived/asr_16k_mono/**/*.wav
```

JSONL의 `text`는 승인된 `surface_text`이며 `normalized_meaning`은 학습
target으로 내보내지 않는다. 반복 문장은 같은 split 안에서는 의도적인
다화자·다환경 수집인지 경고하고, train/validation을 가로지르면 차단한다.

10개 사람 검수가 끝나면 자동 Baseline Report와 prediction별 최신 review
revision을 합쳐 JSON과 Markdown 보고서를 만든다.

```bash
python3 ~/.codex/skills/use-busan-project-venv/scripts/run.py \
  busan-lab finalize-report \
  --experiment-id task-002-nvidia-korean-conformer-ctc-pretrained-v0
```

검수 누락, 불확실 판정의 메모 누락, 알 수 없는 오류 유형, experiment와
prediction 불일치가 있으면 보고서를 생성하지 않는다. 자동 지표는 사람 판정으로
덮어쓰지 않고 `automatic_metrics`와 `human_review`로 분리해 기록한다.

TASK-003A 재평가는 기존 Prediction과 최신 Human Review를 변경하지 않고,
버전이 있는 calibration profile을 사용해 별도 JSON/Markdown revision을 만든다.

```bash
python3 ~/.codex/skills/use-busan-project-venv/scripts/run.py \
  busan-lab calibrate-evaluation \
  --experiment-id task-002-nvidia-korean-conformer-ctc-pretrained-v0 \
  --profile data/lab/calibrations/task-003a-surface-asr-evaluation-v1.json
```

보정 계약은 관찰 가능한 `observed_error`와 확정되지 않은
`suspected_cause`를 분리한다. `candidate` 상태의 허용 변이는 실패나 보존으로
강제 판정하지 않으며, 사람 검수에서 승인되기 전까지 불확실 변이로 집계한다.

## 지표 정의

- CER: NFC 정규화 후 공백·문장부호만 제외한 한글 음절 문자 기준
- Dialect Preservation Rate: 등록된 Surface 부산 표현이 가설에 그대로 남은 비율
- Context Overcorrection Rate: Surface 표현은 사라지고 등록된 표준화 후보가 나온 비율
- High-confidence Wrong: confidence가 지원되고 `CER > 0`이며 기본 0.85 이상
- p50/p95 latency: adapter가 기록한 발화별 추론 시간

`severity`와 `confidence`를 합치지 않는 원칙에 따라 confidence는 오류 크기가
아니라 모델 판단의 위험 신호로만 사용한다.

## 실험·예측·검수·A/B 비교

연구 UI의 모델 평가에는 `experiment_id`, 모델 이름·버전, 선택적 latency를
입력한다. 평가 결과는 응답으로만 반환되지 않고 다음 계보로 영속 저장된다.

```text
ExperimentRun
→ StoredPrediction
→ EvaluationCaseResult
→ HumanReview revision
```

같은 `experiment_id`에는 같은 모델과 실험 조건만 사용할 수 있다. 같은 발화의
저장된 예측 두 개는 UI 또는 `POST /api/comparisons`에서 CER, 부산 표현 보존율,
과보정률, confidence 변화량으로 비교할 수 있다.

자동 오류 분류는 확정 원인이 아니라 `automatic_failure_candidates`다. 사람은
`confirmed`, `rejected`, `uncertain` 중 하나와 오류 유형·근거 메모를 append-only
검수 기록으로 남긴다.

주요 API:

```text
GET/POST /api/experiments
GET/POST /api/predictions
GET/POST /api/reviews
GET      /api/review-queue
POST     /api/comparisons
```

## 테스트와 정적 검증

```bash
RUNNER=~/.codex/skills/use-busan-project-venv/scripts/run.py
python3 "$RUNNER" pytest
python3 "$RUNNER" ruff check src/busan_lab tests
python3 "$RUNNER" mypy --no-incremental --cache-dir /tmp/busan-lab-mypy src/busan_lab
python3 "$RUNNER" busan-lab export-schemas --output reports/schemas
```

테스트는 다음을 포함한다.

- 정상: 44.1kHz stereo WAV 원본 → 48kHz stereo PCM24 마스터
- 정상: 48kHz mono/stereo WAV 입력의 마스터 채널 보존
- 정상: 마스터 → ASR 16k PCM16, 발음 24k PCM16, TTS 48k PCM24 WAV
- 실패: 디코딩 불가 파일, 저장 동의 없는 persistent upload
- 경계: candidate/approved 표현, 표현 없는 발화, 빈 reference 처리
- 회귀: `주이소 → 주세요` 과보정과 높은 confidence 오답 export
- 회귀: 정확 일치·누락·표준어 변경·음향 치환·띄어쓰기·후보 변이 판정
- 누수: 동일 화자와 동일 오디오가 split을 넘는 manifest 거부
- 학습 누수: frozen Benchmark와 발화·화자·오디오 lineage·동일 Surface 차단
- 학습 계약: 동의·검수·음질·train/validation과 모델 중립 ZIP 검증
- 통합: 업로드 → 분석 → 평가 → 오류 export → benchmark 고정

## JSON Schema

```bash
python3 ~/.codex/skills/use-busan-project-venv/scripts/run.py \
  python scripts/export_schemas.py
```

생성 위치:

- `reports/schemas/utterance.schema.json`
- `reports/schemas/benchmark-manifest.schema.json`
- `reports/schemas/evaluation-case.schema.json`
- `reports/schemas/experiment-run.schema.json`
- `reports/schemas/stored-prediction.schema.json`
- `reports/schemas/precomputed-prediction.schema.json`
- `reports/schemas/human-review.schema.json`
- `reports/schemas/label-revision.schema.json`
- `reports/schemas/prediction-comparison.schema.json`
- `reports/schemas/human-reviewed-baseline.schema.json`
- `reports/schemas/evaluation-calibration-profile.schema.json`
- `reports/schemas/evaluation-calibration-report.schema.json`
- `reports/schemas/training-dataset.schema.json`
- `reports/schemas/training-dataset-validation-report.schema.json`
- `reports/schemas/training-export-record.schema.json`
- `reports/schemas/training-split-assignments.schema.json`
- `reports/schemas/training-recording-import-plan.schema.json`
- `reports/schemas/training-recording-import-manifest.schema.json`

Gate 2 감사에서 추가된 strict schema는 `artifacts/gate-2/schemas/`에 별도
snapshot으로 보존한다. Benchmark identity, artifact 복구, blinded A/B, 독립 Test,
Gate criteria/evidence/assessment 계약을 포함한다.

## 알려진 한계

- TASK-002 RIVA Baseline, TASK-003A 평가 보정과 TASK-003B Nemotron
  pretrained Baseline은 완료됐다. TASK-003C에서 명확한 음성 3개의 빈 출력이
  Adapter 오류가 아니라 RNNT blank-only 모델 출력임을 확인했다. 모델 내부의
  음향적 원인은 아직 설명하지 못한다.
- 24-bit/float/RF64 등 모든 WAV 변형을 검증한 것은 아니며 현재 자동 테스트는
  PCM16 입력 WAV를 기준으로 한다.
- 현재 F0는 시각 검사용 autocorrelation baseline이며 Prosody 판정 모델이 아니다.
- log-Mel/F0 분석은 브라우저 응답 크기와 CPU 시간을 위해 첫 60초로 제한된다.
- 오디오 품질 임계값은 고정 baseline이며 실제 데이터 분포로 calibration하지 않았다.
- CER는 한국어 음절 코드포인트 기준이고 형태소/음소 오류를 설명하지 않는다.
- 부산 표현 정답은 사용자가 입력하며 기본 상태는 `candidate`다.
- 로컬 단일 프로세스 도구라 multi-worker 파일 append locking은 아직 없다.
- 개인정보 암호화·삭제 워크플로·보관 기간 enforcement는 production 범위다.

## 사용자가 이해해야 할 개념

- 업로드 원본: 사용자가 제공한 파일 바이트를 수정 없이 보존
- 마스터: 후속 파생본을 다시 만들 수 있는 48kHz mono/stereo PCM24 WAV
- Surface ASR 입력: 마스터에서 만든 16kHz mono PCM16 WAV
- 모델 출력: 정규화하지 않은 `surface_text`, confidence, latency, 모델 버전
- 정답: 사람이 확인한 실제 발화 형태인 `surface_text`
- 최적화 대상: 아직 없음. 현재는 pretrained baseline 측정 단계
- 평가 데이터: frozen manifest에 들어간 speaker/lineage-disjoint 발화
- 실패 계층: DATA/LABEL/AUDIO/TOKENIZER/MODEL/DECODING/LM_BIAS/CALIBRATION 중
  evidence에 따라 분류
- 사용자 노출 가능 여부: 낮은 confidence 또는 candidate 언어 기준은 확정 교정으로
  보여주지 않음

## 현재 작업 순서

```text
TASK-002 RIVA Korean Conformer-CTC Baseline                         완료
TASK-003A Surface ASR Evaluation Calibration                       완료
TASK-003B Nemotron 3.5 Pretrained Surface ASR Baseline Integration 완료(제한 있음)
TASK-003C Nemotron Inference Configuration Validation              완료
TASK-004 Busan ASR Training Dataset Foundation                     완료(200 train + 40 validation)
TASK-005 Nemotron Busan encoder-adapter Pilot                      학습/자동 pilot 완료, Gate 검증 중
Gate 2 closure                                                     FAIL(필수 증거 미완료)
```

다음 실행은 (1) 10개 blinded Human A/B 완료, (2) Train/Validation/기존 Benchmark와
겹치지 않는 다화자 부산 Test 확보, (3) 독립 표준어 Regression 데이터 확보 후 같은
pretrained/adapter로 평가하는 것이다. 기존 40개 Validation은 checkpoint 선택에
사용했으므로 final Test로 재사용하지 않는다.
# ai-apps
# ai-apps
