# GPU PC Codex 복사용 지시문

아래 내용을 Windows GPU PC의 Codex에 그대로 붙여 넣는다.

```text
TASK-005 Nemotron Busan ASR Fine-tuning 준비를 진행해줘.

입력 파일:
- task-005-nemotron-train-pool-v0.zip
- validate_train_pool.py

중요:
- 이 ZIP은 승인된 부산 발화 200개의 Train pool이다.
- 현재 단일 화자뿐이며 독립 Validation은 없다.
- package_metadata.json의 training_permitted=false를 유지해라.
- Validation 없이 Fine-tuning을 시작하거나 성능을 주장하지 마라.
- 원본 transcript, WAV, target_lang=ko-KR을 수정하지 마라.
- busan-surface-v0@1.0.0 Benchmark를 학습/Validation에 사용하지 마라.

1. WSL2 Ubuntu에서 nvidia-smi로 GPU, VRAM, Driver를 확인해라.
2. 다음을 실행해 Train ZIP 무결성을 확인해라.

python3 validate_train_pool.py \
  --package /mnt/c/Users/<Windows-user>/Downloads/task-005-nemotron-train-pool-v0.zip

3. 결과가 train_utterances=200, validation_required=true,
   training_permitted=false인지 확인해라.
4. 공식 NVIDIA NeMo 저장소를 다음 revision으로 고정해라.

git clone https://github.com/NVIDIA-NeMo/Speech.git NeMo
git -C NeMo checkout 6c57e73e83de967eed4d334c493ac313b9afd147

5. 아래 고정 모델 artifact를 사용할 준비만 해라.

model_id: nvidia/nemotron-3.5-asr-streaming-0.6b
revision: f3d333391852ba876df169dcc9ba902d25b6ab0b
artifact: nemotron-3.5-asr-streaming-0.6b.nemo
language: ko-KR

6. 아직 학습은 실행하지 말고 다음을 보고해라.
- GPU 이름과 VRAM
- Windows/WSL2/NVIDIA Driver 버전
- ZIP 검증 결과
- NeMo revision 확인
- 모델 다운로드 가능 여부
- Validation 패키지가 추가되면 실행할 정확한 Fine-tuning 명령 초안

기존 파일을 삭제하거나 결과를 성공으로 가장하지 마라.
```
