# TASK-005 Nemotron Train Pool Handoff

이 디렉터리는 승인된 TASK-004 녹음을 Train 전용 ZIP으로 묶고 외부 GPU PC에서
무결성을 검사한다. 현재 200개는 한 화자뿐이므로 Validation이 추가되기 전에는
Fine-tuning을 시작하지 않는다.

Mac에서 생성:

```bash
python3 ~/.codex/skills/use-busan-project-venv/scripts/run.py \
  python \
  external_training/nemotron_3_5/export_train_pool.py \
  --data-root data/lab \
  --import-id task-004-solo-speaker-001-v0 \
  --expected-count 200 \
  --output ~/Downloads/task-005-nemotron-train-pool-v0.zip
```

Windows WSL2 Ubuntu에서 검사:

```bash
python3 validate_train_pool.py \
  --package /mnt/c/Users/<Windows-user>/Downloads/task-005-nemotron-train-pool-v0.zip
```

고정 모델은 `nvidia/nemotron-3.5-asr-streaming-0.6b` revision
`f3d333391852ba876df169dcc9ba902d25b6ab0b`이다. Fine-tuning은 NVIDIA NeMo
revision `6c57e73e83de967eed4d334c493ac313b9afd147`의 공식
`examples/asr/speech_to_text_finetune.py` 경로를 사용한다.
