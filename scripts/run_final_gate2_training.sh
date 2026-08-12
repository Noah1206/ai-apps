#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEMO_ROOT="${NEMO_ROOT:-/mnt/c/Users/ab409/Downloads/PORTER-main/NeMo}"
PYTHON="${NEMO_PYTHON:-/home/ab409/.venvs/task-005-nemo/bin/python}"
DATA_ROOT="$PROJECT_ROOT/data/lab/gate2/final-attempt-v2"
TRAIN_MANIFEST="$DATA_ROOT/train/manifest.absolute.jsonl"
VALIDATION_MANIFEST="$DATA_ROOT/validation/manifest.absolute.jsonl"
COMMITMENT="$DATA_ROOT/commitment.json"
MODEL="$PROJECT_ROOT/artifacts/gate-2/reproducibility/recovered/nemotron-3.5-asr-streaming-0.6b.nemo"
EXPERIMENT_ROOT="$DATA_ROOT/training"
RUN_LOCK="$EXPERIMENT_ROOT/one-training-job.started"
RUN_VERSION="attempt-1"

for path in "$PYTHON" "$NEMO_ROOT" "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" "$COMMITMENT" "$MODEL"; do
  if [[ ! -e "$path" ]]; then
    echo "missing required path: $path" >&2
    exit 20
  fi
done

"$PYTHON" - "$PROJECT_ROOT" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
data = root / "data/lab/gate2/final-attempt-v2"
commitment = json.loads((data / "commitment.json").read_text(encoding="utf-8"))
if commitment["status"] != "frozen_before_training" or not commitment["one_attempt"]:
    raise SystemExit("invalid final-attempt commitment")

artifacts = {
    "train_manifest_sha256": data / "train/manifest.source.jsonl",
    "validation_manifest_sha256": data / "validation/manifest.source.jsonl",
    "test_v2_manifest_sha256": data / "test-v2/manifest.json",
}
for field, path in artifacts.items():
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != commitment[field]:
        raise SystemExit(f"commitment hash mismatch: {field}")

expected_counts = {
    "train_utterances": 1000,
    "train_speakers": 41,
    "validation_utterances": 140,
    "validation_speakers": 14,
    "test_v2_utterances": 100,
    "test_v2_speakers": 10,
}
if commitment["counts"] != expected_counts:
    raise SystemExit("unexpected frozen split counts")
if any(items for pair in commitment["overlaps"].values() for items in pair.values()):
    raise SystemExit("frozen splits overlap")

model = root / "artifacts/gate-2/reproducibility/recovered/nemotron-3.5-asr-streaming-0.6b.nemo"
expected_model_sha256 = "210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74"
if hashlib.sha256(model.read_bytes()).hexdigest() != expected_model_sha256:
    raise SystemExit("base model hash mismatch")
print("final Gate 2 training preflight passed")
PY

mkdir -p "$EXPERIMENT_ROOT"
if ! mkdir "$RUN_LOCK" 2>/dev/null; then
  if [[ "${TASK005_INFRA_RESTART:-}" != "OOM_CHECKPOINT" ]]; then
    echo "blocked: the one permitted training job has already been started" >&2
    exit 21
  fi
  if find "$EXPERIMENT_ROOT" -type f -name '*.ckpt' -print -quit | grep -q .; then
    echo "blocked: a usable checkpoint already exists" >&2
    exit 22
  fi
  if ! mkdir "$EXPERIMENT_ROOT/one-infrastructure-restart.used" 2>/dev/null; then
    echo "blocked: the one infrastructure restart was already used" >&2
    exit 23
  fi
  RUN_VERSION="attempt-1-oom-recovery"
fi

export PYTHONPATH="$NEMO_ROOT"
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=0
export TASK005_EXPECTED_TRAINABLE_PARAMETERS=1622016
export TASK005_NONFINITE_LOSS_GUARD=YES
export TASK005_AMP_INIT_SCALE=1024

cd "$NEMO_ROOT"
exec "$PYTHON" -c \
  'import runpy; from lightning.pytorch import seed_everything; seed_everything(0, workers=True); runpy.run_path("examples/asr/asr_adapters/train_asr_adapter.py", run_name="__main__")' \
  --config-path=../conf/asr_adapters \
  --config-name=asr_adaptation.yaml \
  model.pretrained_model=null \
  model.nemo_model="$MODEL" \
  ++model.compute_eval_loss=true \
  model.adapter.adapter_name=busan_ko_kr_v0 \
  model.adapter.adapter_type=linear \
  model.adapter.adapter_module_name=encoder \
  model.adapter.linear.in_features=1024 \
  model.adapter.linear.dim=32 \
  model.adapter.linear.dropout=0.0 \
  model.train_ds.manifest_filepath="$TRAIN_MANIFEST" \
  ++model.train_ds.batch_duration=8 \
  ++model.train_ds.quadratic_duration=null \
  ++model.train_ds.use_bucketing=true \
  model.train_ds.num_workers=0 \
  model.train_ds.pin_memory=false \
  model.validation_ds.manifest_filepath="$VALIDATION_MANIFEST" \
  ++model.validation_ds.lang_field=target_lang \
  model.validation_ds.batch_size=1 \
  model.validation_ds.num_workers=0 \
  model.validation_ds.pin_memory=false \
  model.optim.name=adamw \
  model.optim.lr=5e-4 \
  model.optim.weight_decay=0.0 \
  model.optim.sched.name=CosineAnnealing \
  model.optim.sched.warmup_steps=20 \
  model.optim.sched.warmup_ratio=null \
  model.optim.sched.min_lr=1e-5 \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.strategy=auto \
  trainer.precision=16-mixed \
  trainer.sync_batchnorm=false \
  trainer.max_epochs=null \
  trainer.max_steps=600 \
  trainer.val_check_interval=25 \
  trainer.accumulate_grad_batches=8 \
  trainer.gradient_clip_val=1.0 \
  trainer.num_sanity_val_steps=0 \
  trainer.log_every_n_steps=1 \
  exp_manager.exp_dir="$EXPERIMENT_ROOT" \
  exp_manager.name=final-gate2-adapter \
  exp_manager.checkpoint_callback_params.monitor=val_loss \
  exp_manager.checkpoint_callback_params.mode=min \
  exp_manager.checkpoint_callback_params.save_top_k=2 \
  exp_manager.checkpoint_callback_params.always_save_nemo=false \
  ++exp_manager.checkpoint_callback_params.save_best_model=true \
  ++exp_manager.checkpoint_callback_params.save_nemo_on_train_end=true \
  ++exp_manager.version="$RUN_VERSION" \
  exp_manager.resume_if_exists=false \
  exp_manager.resume_ignore_no_checkpoint=false
