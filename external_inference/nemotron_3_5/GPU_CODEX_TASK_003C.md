# GPU PC Codex 복붙용 — TASK-003C Nemotron 빈 출력 진단

아래 내용 전체를 **Windows NVIDIA GPU PC에서 연 Codex**에 그대로
복사해서 전달한다. 이 문서는 셸에서 직접 실행하는 스크립트가 아니라 Codex
작업 지시문이다.

---

현재 작업은 `TASK-003C: Nemotron Inference Configuration Validation`이다.

목표는 TASK-003B에서 빈 문자열이 나온 3개 발화가 실제 모델 출력인지,
오디오·추론 설정·RNNT 반환값 파싱·Adapter 변환 문제인지 확인하는 것이다.
Fine-tuning이나 모델 성능 개선은 하지 않는다.

## 입력

내가 전달한 다음 두 ZIP을 사용한다.

1. 최신 `nemotron_3_5` 외부 GPU 실행 패키지 ZIP
2. Mac Audio Lab에서 Export한 frozen Benchmark
   `busan-surface-v0@1.0.0` ZIP

먼저 두 ZIP의 실제 위치와 압축을 푼 디렉터리를 찾아라. 이전 TASK-003B
결과 디렉터리와 결과 ZIP은 삭제·수정·덮어쓰기 하지 마라. 새 작업 디렉터리와
새 출력 디렉터리를 사용하라.

## 진단 대상

reference transcript는 사람이 결과를 판단하기 위한 표기일 뿐이며 모델
입력, prompt, context, lexicon, boosting 또는 후처리에 절대 사용하지 마라.
실제 대상 선택은 아래 `utterance_id`로만 한다.

| 역할 | utterance_id | 사람이 확인할 reference |
|---|---|---|
| 빈 출력 재현 1 | `3faa344e-968d-42cb-baf8-f1847a936a98` | 와따 맛있노 |
| 빈 출력 재현 2 | `75b96c24-cab0-4e52-839b-dd67b496e58a` | 내일 같이 가재이 |
| 빈 출력 재현 3 | `c95d2f85-a98a-4c19-9476-e55b92a49dc0` | 그거 아이다 |
| 정상 통제 | `38f63a59-325b-4e53-9287-ab094c6a889d` | 지금 뭐 하노 |

Benchmark 자체는 10개 전체 계약을 먼저 검증하되, TASK-003C 상세 진단은 위
4개만 수행한다.

## 고정 모델 계약

다음 값을 임의로 변경하지 마라.

```text
model_id:
  nvidia/nemotron-3.5-asr-streaming-0.6b

requested_revision:
  f3d333391852ba876df169dcc9ba902d25b6ab0b

processor language:
  ko-KR

model family:
  FastConformer-RNNT

decoder:
  RNNT

fine_tuned:
  false
```

기존 패키지에 고정된 환경과 API를 그대로 사용한다. 모델 revision,
Transformers/NeMo 경로, language, decoding, streaming chunk 또는
후처리 설정을 임의로 바꿔 결과가 좋아지게 만들지 마라.

금지 항목:

- Fine-tuning checkpoint
- reference transcript conditioning
- context biasing 또는 word boosting
- 외부 language model
- LLM 또는 사전 기반 후처리
- 빈 결과를 사람이 채워 넣기
- RNNT token probability를 임의의 발화 confidence로 만들기
- 실제 반환값을 보지 않고 결과를 추측하거나 꾸미기

## 실행 전에 할 일

1. 최신 패키지의 `README.md`, 설정 파일, 진단 코드와 테스트를 먼저 읽어라.
2. `python run_diagnostics.py --help`로 **실제로 제공된 옵션**을 확인하라.
3. TASK-003C의 고정 진단 CLI는 `run_diagnostics.py`다. 옵션을 임의로
   추가하거나 `run_inference.py`로 대체하지 마라.
4. 패키지의 CPU 계약 테스트를 먼저 실행하라.
5. WSL2/Docker 환경에서 `nvidia-smi`,
   `torch.cuda.is_available()`, GPU 이름, VRAM, Driver, CUDA, PyTorch,
   Transformers 버전을 기록하라.
6. 기존 TASK-003B와 같은 고정 revision이 resolve되는지 확인하라. 다르면
   추론하지 말고 중단해 보고하라.

기존 Phase B의 기본 계약 검증 명령은 다음 형태다. 실제 경로만 맞춰라.

```bash
python validate_package.py \
  --benchmark-package /workspace/busan-surface-v0--1.0.0.zip \
  --config config.example.yaml
```

## 반드시 수행할 진단

TASK-003C 진단 CLI는 위 4개 ID를 코드의 고정 allowlist로 자동 선택한다.
별도의 transcript나 ID filter를 전달하지 말고 다음 명령으로 한 번의 고정
설정 Run을 실행하라.

```bash
python run_diagnostics.py \
  --benchmark-package /workspace/busan-surface-v0--1.0.0.zip \
  --config config.example.yaml \
  --output-dir /workspace/task-003c-output
```

`/workspace/task-003c-output`이 이미 있으면 덮어쓰지 말고 timestamp가
붙은 새 output 경로로 같은 명령을 실행하라.

각 발화에서 최소한 다음을 확인·저장하라.

### 1. Audio probe

- Benchmark의 audio relative path
- SHA-256
- sample rate
- channel 수
- sample width/PCM 형식
- frame 수와 duration
- waveform sample 수
- peak absolute amplitude
- RMS
- non-finite sample 존재 여부
- 완전 무음 여부

WAV를 resample하거나 normalize해 결과를 바꾸지 마라. 계약 검증용 원본
16 kHz mono PCM16 waveform을 그대로 모델에 전달하라.

### 2. 모델 원본 반환값

- 실제 Python 반환 타입
- `.sequences` 존재 여부
- sequence shape, dtype, device
- 실제 token ID sequence
- 특수 토큰을 제거하기 전과 후의 decode 결과
- 공식 API가 함께 반환한 안전하게 직렬화할 수 있는 필드

Tensor나 모델 객체 전체를 JSON에 억지로 넣지 말고, 패키지에서 정의한
안전한 요약 형식으로 저장하라. 민감한 토큰, 사용자 홈 경로 또는 모델
weight 자체는 출력에 넣지 마라.

### 3. Adapter 전후 추적

- processor에 전달한 sample rate
- processor language 값
- processor 출력 key/shape
- 모델 `generate`에 전달한 key
- 원본 sequence
- processor decode 결과
- Adapter가 반환한 transcript
- 최종 `surface_text`
- 빈 문자열이 된 단계

Streaming chunk 설정이 이 Offline Run에 개입하지 않았는지, 공식 Offline
`generate` 경로가 사용됐는지도 명시하라.

### 4. 재현성

`run_diagnostics.py`가 저장한 재현성 정보와 raw trace를 확인하라. 추가
재실행이 필요하면 서로 다른 **새 출력 디렉터리**에 위와 동일한 명령을
실행하라. 첫 결과를 덮어쓰지 마라.

## 필요한 출력

최신 패키지가 정한 정확한 파일명을 우선하되, 결과에는 최소한 다음 증거가
있어야 한다.

```text
audio_probe.json
raw_model_output.jsonl
adapter_trace.jsonl
task_003c_summary.json
run.log
```

GPU PC에서 Mac으로 반송할 필수 결과는 위 5개다. 추가 재현 Run이 필요했다면
별도 하위 디렉터리로 구분해 함께 보존하되, 기존 TASK-003B 4개 결과 파일을
섞거나 수정하지 마라.

`task_003c_summary.json` 또는 동등한 summary에는 다음을 명확히 기록하라.

- 대상 4개 ID가 모두 처리됐는지
- 빈 출력 3개가 원본 모델 sequence/decode 단계부터 비었는지
- Adapter에서만 비게 됐는지
- 정상 통제 문장이 정상으로 유지됐는지
- 두 Run이 동일했는지
- audio contract 이상 여부
- language와 Offline API 확인 결과
- 최종 원인 분류:
  - `MODEL_RETURNED_EMPTY`
  - `ADAPTER_EXTRACTION_ERROR`
  - `AUDIO_CONTRACT_ERROR`
  - `INFERENCE_CONFIGURATION_ERROR`
  - `NOT_REPRODUCED`
  - `UNRESOLVED`
- 근거와 아직 GPU에서 확인하지 못한 항목

근거가 부족하면 반드시 `UNRESOLVED`로 남겨라. 빈 출력이 확인됐다는 이유만으로
원인을 임의로 모델 성능 또는 Adapter 오류로 단정하지 마라.

## 테스트와 검증

1. 최신 패키지 테스트를 모두 실행하라.
2. 다음 명령으로 진단 결과를 검증하라.

```bash
python validate_diagnostics.py \
  --diagnostics-dir /workspace/task-003c-output
```

   timestamp가 붙은 다른 output 디렉터리를 사용했다면 위 경로만 실제 경로로
   바꿔라.
3. validator가 없거나 실패하면 이를 숨기지 말고 정확한 명령, exit code,
   오류를 보고하라.
4. 4개 중 하나라도 누락됐으면 성공으로 보고하지 마라.
5. JSON/JSONL이 parse 가능한지 확인하고, utterance ID 중복·누락도 확인하라.
6. 기존 TASK-003B `predictions.jsonl`이나 사람 검수 보고서를 수정하지 마라.

TASK-003C는 원인 진단이다. 기존 Audio Lab Import용 TASK-003B
`predictions.jsonl`을 새 진단 결과로 대체하지 마라. 추론 코드/설정 오류가
확인되더라도 이 작업에서 기존 10개 결과를 덮어쓰지 말고, 수정안과 별도의
v2 재추론 필요 여부만 보고하라.

## Mac으로 보낼 ZIP

검증을 통과한 아래 TASK-003C 결과 5개만 새 ZIP에 넣어라.

```text
audio_probe.json
raw_model_output.jsonl
adapter_trace.jsonl
task_003c_summary.json
run.log
```

모델 캐시, weight, Benchmark 음성, Hugging Face token, 사용자 홈 경로,
기존 TASK-003B의 `predictions.jsonl`·`execution_metadata.json`·
`inference_summary.json`·`run.log`는 ZIP에 넣지 마라.

WSL2에서 출력 디렉터리의 상위 위치로 이동한 뒤, 실제 출력 디렉터리 이름을
사용해 다음과 같은 새 파일을 만든다.

```bash
python - <<'PY'
from pathlib import Path
import zipfile

source = Path("/workspace/task-003c-output")
target = Path("/workspace/task-003c-nemotron-diagnostics-results.zip")
required = (
    "audio_probe.json",
    "raw_model_output.jsonl",
    "adapter_trace.jsonl",
    "task_003c_summary.json",
    "run.log",
)

if not source.is_dir():
    raise SystemExit(f"missing diagnostic output: {source}")
if target.exists():
    raise SystemExit(f"refusing to overwrite: {target}")
missing = [name for name in required if not (source / name).is_file()]
if missing:
    raise SystemExit(f"missing diagnostic result files: {missing}")

with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
    for name in required:
        path = source / name
        archive.write(path, Path(source.name) / name)

with zipfile.ZipFile(target) as archive:
    bad = archive.testzip()
    if bad is not None:
        raise SystemExit(f"corrupt ZIP member: {bad}")
    print("\n".join(archive.namelist()))

print(target)
PY
```

추가 재현 Run을 별도 디렉터리에서 실행했다면 그 Run은 별도 ZIP으로 보존하고,
위 기본 결과 5개 ZIP과 섞지 마라. 기존 결과를 이동하거나 삭제하지 마라.

최종 ZIP의 SHA-256도 계산하라.

```bash
sha256sum /workspace/task-003c-nemotron-diagnostics-results.zip
```

Windows Explorer에서 사용자가 찾을 수 있는 경로로 ZIP을 복사하되 기존
파일을 덮어쓰지 마라. 예:

```text
C:\Users\<사용자>\Downloads\task-003c-nemotron-diagnostics-results.zip
```

WSL 경로와 Windows 경로를 둘 다 사용자에게 알려라.

## 최종 보고 형식

작업을 마치면 한국어로 다음을 보고하라.

1. 사용한 모델 ID와 requested/resolved revision
2. GPU/Driver/CUDA/PyTorch/Transformers 환경
3. 실제 사용한 진단 명령
4. 대상 4개의 `utterance_id`
5. 발화별 audio probe 요약
6. 발화별 원본 output type, token ID, raw decode, Adapter transcript
7. 빈 출력이 발생한 정확한 단계
8. 반복 실행 재현 여부
9. 진단 결과 validator와 테스트 결과
10. 원인 판정과 근거
11. 실패하거나 아직 확인하지 못한 항목
12. 생성한 결과 ZIP의 WSL 경로, Windows 경로, SHA-256

실제로 실행하지 않은 작업, 확인하지 않은 반환값, 생성되지 않은 파일을
완료했다고 주장하지 마라. 문제가 생기면 수동으로 결과를 채우지 말고 로그와
증거를 보존한 뒤 차단 요소를 보고하라.
