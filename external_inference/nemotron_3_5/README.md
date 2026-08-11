# TASK-003B Nemotron 3.5 External GPU Inference

이 디렉터리와 Mac Audio Lab이 Export한 Benchmark ZIP만 외부 NVIDIA GPU
컴퓨터로 복사하면 된다. Audio Lab 저장소 전체를 복사하지 않는다.

이 패키지는 `nvidia/nemotron-3.5-asr-streaming-0.6b`의 고정 pretrained
revision을 순수 Offline 방식으로 추론해 Audio Lab 호환 `predictions.jsonl`을
만든다. Fine-tuning, reference conditioning, context biasing, word boosting,
외부 언어 모델과 LLM 후처리는 하지 않는다.

## 고정 계약

| 항목 | 고정값 또는 상태 |
|---|---|
| Experiment | `task-003b-nemotron-3.5-asr-streaming-0.6b-pretrained-v0` |
| Model ID | `nvidia/nemotron-3.5-asr-streaming-0.6b` |
| Revision | `f3d333391852ba876df169dcc9ba902d25b6ab0b` |
| Model version | `nemotron-3.5-asr-streaming-0.6b-v1` |
| License | OpenMDW-1.1 |
| Access | 공개, Hugging Face gated model 아님 |
| Architecture | Cache-Aware FastConformer-RNNT with language prompt |
| Model class | `Nemotron3_5AsrForRNNT` via `AutoModelForRNNT` |
| Processor | `Nemotron3_5AsrProcessor` via `AutoProcessor` |
| Tokenizer | `ParakeetTokenizer` |
| Language | processor `language="ko-KR"` |
| Audio | 16 kHz, mono, uncompressed PCM16 WAV |
| Decoding | Transformers 5.13 default greedy RNNT, max 10 symbols/frame |
| Output | native punctuation/capitalization을 포함한 raw transcript |
| Confidence | `null`, `confidence_supported=false` |
| Benchmark | `busan-surface-v0@1.0.0`, frozen, 10개 |

공식 model card는 NeMo 26.06과 NeMo `ASRModel.from_pretrained`도 제시한다.
그러나 이번 패키지는 반환 객체와 transcript 추출 절차가 공식 예제로 명확한
Transformers 5.13.0 Offline RNNT 경로를 사용한다.

```python
inputs = processor(audio, sampling_rate=16000, language="ko-KR", return_tensors="pt")
output = model.generate(**inputs, return_dict_in_generate=True)
transcript = processor.decode(output.sequences, skip_special_tokens=True)
```

실행 코드는 반환값을 문자열로 가정하지 않는다. `.sequences`가 없거나
decode 결과가 문자열이 아니면 즉시 실패하고, 실제 반환 객체의 클래스 이름은
`execution_metadata.json`의 `generation_output_type`에 기록한다.

## 확인된 정보와 GPU에서 남은 확인

확인된 공식 정보:

- Python 3.11 이상 권장
- Transformers 지원은 5.13.0부터
- processor sample rate는 16,000 Hz
- `ko-KR`은 transcription-ready locale이며 prompt dictionary의 공식 키
- punctuation과 capitalization은 모델의 native output
- `model.safetensors` 크기는 2,552,062,944 bytes
- 공식 성능 자료에 사용된 GPU는 NVIDIA H100
- NVIDIA가 열거한 지원 microarchitecture는 Turing, Volta, Ampere, Lovelace,
  Hopper, Blackwell, Jetson

아직 외부 GPU에서 확인하지 않은 항목:

- 사용자의 실제 GPU에서 필요한 최소 VRAM과 최대 길이
- 선택한 전체 버전 조합의 실제 import/load 성공
- 실제 `generation_output_type`
- 10개 발화의 처리 시간과 RTF
- 공식적으로 의미가 정의된 발화 수준 confidence

NVIDIA는 이 600M 모델의 최소 VRAM 수치를 공개하지 않았다. 따라서 이 저장소는
근거 없는 GB 하한을 통과 조건으로 만들지 않는다. 모델이 GPU에 완전히 올라가지
않으면 Run을 중단하고 GPU/VRAM 정보를 보고한다. 모델 파일, 컨테이너와 캐시를
위해 최소 10 GB의 여유 디스크를 준비하는 것은 이 패키지의 운영 권장치이며
NVIDIA 공식 최소 사양은 아니다.

## 권장 환경: Windows + WSL2 + NVIDIA Docker

네이티브 Windows는 이 패키지의 검증 대상이 아니다. NVIDIA 공식 통합 정보는
Linux를 기준으로 하므로 Windows PC에서는 WSL2 Ubuntu 또는 Linux와 NVIDIA
Container Toolkit을 사용한다.

선택한 재현 환경:

```text
nvcr.io/nvidia/pytorch:26.06-py3
Ubuntu 24.04
Python 3.12
PyTorch 2.13.0a0+8145d630e8
CUDA 13.3.0
Transformers 5.13.0
```

NVIDIA 26.06 PyTorch 컨테이너가 PyTorch와 CUDA를 제공한다.
`requirements.txt`는 이를 덮어쓰지 않는다. 이 조합은 공식 구성요소를
고정한 GPU 실행 후보이며, 프로젝트 자체의 실제 GPU 검증은 아직 남아 있다.

### 1. Windows/WSL2 준비

1. Windows NVIDIA Driver, WSL2 Ubuntu, Docker Desktop의 WSL2 backend와
   NVIDIA Container Toolkit을 준비한다.
2. WSL2 Ubuntu에서 확인한다.

```bash
nvidia-smi
docker run --rm --gpus all nvcr.io/nvidia/cuda:13.3.0-base-ubuntu24.04 nvidia-smi
```

NGC 컨테이너 pull에 인증이 필요한 환경에서는 NVIDIA NGC 공식 절차로 로그인한다.
토큰을 이 디렉터리, config, 명령 기록 또는 출력 파일에 저장하지 않는다.
Hugging Face 모델 자체는 현재 공개·ungated라 모델 다운로드용 HF token은
필수가 아니다.

### 2. 파일 복사

외부 PC의 한 폴더에 다음 두 항목을 둔다.

```text
work/
├── nemotron_3_5/                  # 이 디렉터리 전체
└── busan-surface-v0--1.0.0.zip  # Mac Audio Lab Export
```

실제 Export ZIP 구조는 다음 계약이다.

```text
benchmark.json
derived/asr_16k_mono/**/*.wav
```

별도 `manifest.jsonl` 규격을 만들지 않는다.

### 3. 컨테이너 시작과 의존성 설치

WSL2의 `work/`에서:

```bash
docker run --rm -it --gpus all \
  -v "$PWD:/workspace" \
  -w /workspace/nemotron_3_5 \
  nvcr.io/nvidia/pytorch:26.06-py3 bash
```

컨테이너 안에서:

```bash
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pip install -r requirements.txt
```

다른 PyTorch wheel을 추가 설치하지 않는다.

### 4. Benchmark만 먼저 검증

```bash
python validate_package.py \
  --benchmark-package /workspace/busan-surface-v0--1.0.0.zip \
  --config config.example.yaml
```

검증 항목:

- Benchmark ID/version/frozen/발화 수
- 중복 utterance ID와 중복 오디오 경로
- ZIP 경로 탈출, 절대 경로, backslash, symlink
- Manifest와 실제 ZIP 파일의 정확한 일치
- WAV 존재·CRC·SHA-256
- 16 kHz/mono/PCM16/비어 있지 않은 오디오

reference `surface_text`는 검증된 manifest 안에 남아 있지만 Adapter로 전달하지
않으며 prompt, 보정, boosting에 사용하지 않는다.

같은 검증은 추론 CLI에서도 가능하다.

```bash
python run_inference.py \
  --benchmark-package /workspace/busan-surface-v0--1.0.0.zip \
  --config config.example.yaml \
  --output-dir /workspace/task-003b-validation \
  --validate-only
```

### 5. 실제 추론

새 output 디렉터리를 사용한다.

```bash
python run_inference.py \
  --benchmark-package /workspace/busan-surface-v0--1.0.0.zip \
  --config config.example.yaml \
  --output-dir /workspace/task-003b-output
```

모델 저장소의 요청 revision과 Hugging Face가 돌려준 resolved revision이 다르면
모델을 로드하지 않는다. `model.safetensors` SHA-256도 실행 시 계산한다.

첫 발화 전에 1초 무음 synthetic warm-up을 한 번 수행한다. Benchmark 음성을
warm-up에 사용하지 않는다. 발화별 latency는 CUDA synchronize 전후의 시간이며,
모델 다운로드와 load 시간은 포함하지 않는다.

공식 문서는 batch workload를 지원한다고 설명하지만 공개 Offline 예제의 정확한
batch 반환 계약은 고정하지 않는다. 10개 Pilot에서는 재현 가능한 발화별 latency와
오류 격리를 위해 한 발화씩 처리한다.

### 6. 출력 검증

```bash
python validate_predictions.py \
  --predictions /workspace/task-003b-output/predictions.jsonl \
  --benchmark-package /workspace/busan-surface-v0--1.0.0.zip \
  --schema schemas/predictions.schema.json \
  --config config.example.yaml
```

성공 출력:

```text
task-003b-output/
├── predictions.jsonl
├── execution_metadata.json
├── inference_summary.json
└── run.log
```

출력은 append-only다. 같은 output 디렉터리에 결과 파일이 이미 있으면
덮어쓰지 않고 중단한다. 한 발화라도 실패하면 가능한 Prediction과 실패 Summary를
보존하되 프로세스는 성공 코드로 끝나지 않는다.

### 7. Mac으로 가져갈 파일

네 파일을 모두 Mac으로 가져온다.

```text
predictions.jsonl
execution_metadata.json
inference_summary.json
run.log
```

Audio Lab Import에는 `predictions.jsonl`을 사용한다. 나머지 세 파일은
TASK-003B 실행 증거와 문제 진단용으로 함께 보존한다.

## TASK-003C: 빈 출력 원인 진단

TASK-003B에서 빈 문자열이 나온 세 발화와 정상 출력 통제 발화 한 개를 고정해
원본 RNNT sequence부터 Adapter 출력까지 추적한다. 이 명령은 모델 성능을 다시
평가하거나 `predictions.jsonl`을 만드는 명령이 아니다.

```bash
python run_diagnostics.py \
  --benchmark-package /workspace/busan-surface-v0--1.0.0.zip \
  --config config.example.yaml \
  --output-dir /workspace/task-003c-output
```

진단 대상은 코드에 다음 네 `utterance_id`로 고정되어 있다.

```text
3faa344e-968d-42cb-baf8-f1847a936a98  # 기존 빈 출력
75b96c24-cab0-4e52-839b-dd67b496e58a  # 기존 빈 출력
c95d2f85-a98a-4c19-9476-e55b92a49dc0  # 기존 빈 출력
38f63a59-325b-4e53-9287-ab094c6a889d  # 기존 정상 통제
```

Benchmark 전체 10개 계약을 먼저 검증하지만 상세 추론은 위 네 파일만 수행한다.
reference transcript는 대상 선택이나 모델 입력에 사용하지 않는다. 기존
TASK-003B 결과와 다른 새 output 디렉터리를 사용해야 한다.

출력:

```text
task-003c-output/
├── audio_probe.json
├── raw_model_output.jsonl
├── adapter_trace.jsonl
├── task_003c_summary.json
└── run.log
```

`audio_probe.json`은 SHA-256, 16 kHz mono PCM16 계약, 길이, sample 수, peak,
RMS와 무음 여부를 기록한다. `raw_model_output.jsonl`은 실제 반환 타입과
RNNT token ID를, `adapter_trace.jsonl`은 특수 토큰 제거 전·후 decode와
Adapter 결과를 기록한다. Offline 호출에서는 streaming processor flag,
streamer와 chunk generator를 전달하지 않았다는 사실도 함께 남긴다.

결과를 Mac으로 보내기 전에 검증한다.

```bash
python validate_diagnostics.py \
  --diagnostics-dir /workspace/task-003c-output
```

검증기는 다섯 파일, 고정 ID 네 개의 중복·누락, 오디오 계약, `ko-KR`,
Offline 호출 기록과 완전한 summary를 검사한다. 빈 transcript 자체는 허용한다.
원인은 GPU 증거를 사람이 확인한 뒤 다음 중 하나로 판정한다.

```text
MODEL_RETURNED_EMPTY
ADAPTER_EXTRACTION_ERROR
AUDIO_CONTRACT_ERROR
INFERENCE_CONFIGURATION_ERROR
NOT_REPRODUCED
UNRESOLVED
```

근거가 부족하면 `UNRESOLVED`로 남긴다. 설정 또는 Adapter 오류가 확인되더라도
기존 TASK-003B 결과를 덮어쓰지 않고 별도 v2 재추론을 준비한다.

## Docker 없이 WSL2 venv를 쓰는 경우

가능하지만 현재 재현 기준은 아니다. Python 3.11 이상과 NVIDIA Driver를 준비하고,
PyTorch 공식 설치 선택기에서 실제 Driver와 맞는 CUDA build를 선택한 뒤
`requirements.txt`를 설치한다. 정확한 PyTorch/CUDA 조합을
`execution_metadata.json`으로 기록하고 Docker 결과와 섞어 비교하지 않는다.

```bash
python3 -m venv .venv
source .venv/bin/activate
# PyTorch는 pytorch.org의 현재 공식 명령으로 먼저 설치
python -m pip install -r requirements.txt
```

## 흔한 실패

- `torch.cuda.is_available() == False`: WSL2 GPU passthrough, Windows Driver,
  Docker `--gpus all`을 확인한다.
- CUDA/Driver 불일치: `nvidia-smi`와 선택한 26.06 컨테이너 요구조건을 확인한다.
- 컨테이너 pull 실패: NGC 접근/로그인을 확인한다.
- Hugging Face download 실패: 네트워크, 프록시, 디스크 공간을 확인한다.
- revision mismatch: `config.example.yaml`을 바꾸지 말고 실행을 중단한다.
- ZIP hash/CRC/WAV 오류: Mac에서 Benchmark를 다시 Export한다. reference나
  Prediction을 수동으로 고쳐 우회하지 않는다.
- OOM: 실패 로그와 GPU VRAM을 보존한다. CPU offload나 quantization을 임의로
  켜서 같은 Baseline으로 보고하지 않는다.
- 일부 발화 실패: 새 output 디렉터리로 원인을 해결한 뒤 전체 10개를 다시
  실행한다. 누락된 결과를 수동으로 채우지 않는다.

## 환경 정리

컨테이너는 `--rm`으로 종료 시 삭제된다. 필요할 때만 사용자가 직접 다음 캐시와
이미지를 정리한다.

```bash
rm -rf /workspace/nemotron_3_5/model-cache
docker image rm nvcr.io/nvidia/pytorch:26.06-py3
```

`task-003b-output`은 Mac에 안전하게 복사하고 검증하기 전에는 삭제하지 않는다.

## 공식 자료

- [NVIDIA Nemotron 3.5 ASR model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
- [Transformers 5.13 Nemotron 3.5 ASR API](https://huggingface.co/docs/transformers/v5.13.0/model_doc/nemotron3_5_asr)
- [NVIDIA PyTorch 26.06 release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-06.html)
- [NVIDIA Speech NIM model card](https://build.nvidia.com/nvidia/nemotron-asr-streaming/modelcard)
- [OpenMDW 1.1 license](https://openmdw.ai/license/1-1/)
