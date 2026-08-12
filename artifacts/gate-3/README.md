# Gate 3 — Streaming ASR

Gate 3 is **FAIL (28/29 frozen checks passed)**. The repository has a model-neutral streaming
event contract, stable-prefix tracker, lifecycle tests, an accepted RTX 2070 cache-aware smoke,
and three frozen 20-case/10-speaker real-time-paced engineering benchmarks. Gate 4 has not started.

## Tracked evidence

- `status/GATE3_STATUS_2026-08-12.md`: current decision and remaining blockers
- `status/FINAL_GATE3_RUNTIME_V5_ASSESSMENT_2026-08-12.md`: current final frozen assessment
- `status/FINAL_GATE3_RUNTIME_V4_ASSESSMENT_2026-08-12.md`: preserved v2/runtime-v4 assessment
- `status/FINAL_GATE3_ASSESSMENT_2026-08-12.md`: preserved v1/runtime-v3 assessment
- `status/GATE3_EVALUATION_PROTOCOL_V2_2026-08-12.md`: runtime-v4 precommitted protocol
- `status/gate3-criteria.v2.json`: runtime-v4 machine-readable frozen thresholds
- `status/GATE3_EVALUATION_PROTOCOL_V3_2026-08-12.md`: runtime-v5 precommitted protocol
- `status/gate3-criteria.v3.json`: runtime-v5 machine-readable frozen thresholds
- `status/GATE3_EVALUATION_PROTOCOL_2026-08-12.md`: evaluation rules to freeze before the
  multi-sample run
- `evaluation-results/smoke-001/summary.json`: transcript-redacted accepted smoke evidence
- `evaluation-results/engineering-v1/summary.json`: transcript-redacted final batch evidence
- `evaluation-results/engineering-v1/attempt-history.json`: immutable candidate history
- `evaluation-results/engineering-v2/summary.json`: transcript-redacted runtime-v4 evidence
- `evaluation-results/engineering-v3/summary.json`: transcript-redacted runtime-v5 evidence

Licensed audio, raw transcripts, model weights, full per-chunk events, and runtime logs stay
under ignored `data/lab/gate3/`. The accepted local source is `smoke-v3`; `smoke-v0` and
`smoke-v1` are failed integration attempts, and `smoke-v2` is excluded because two WSL child
processes wrote the same output directory.

The current final batch source is `engineering-v3-run-001`. Runtime-v5 immediately processes the
synthetic flush after EOF and passed finalization-lag p95 at 246.17ms, but raw exact agreement was
13/20 (0.65) against the frozen 19/20 minimum. The criteria were not relaxed and the v3 batch was
not repeated. Earlier v1/v2 runs remain immutable failure history.

## Reproduce the accepted smoke

Run from WSL with the frozen Python 3.13.14 NeMo environment:

```bash
cd /mnt/c/Users/ab409/orca/workspaces/ai-apps/wreckfish
/home/ab409/.venvs/task-005-nemo/bin/python \
  external_inference/nemotron_3_5/run_gate3_streaming.py \
  --nemo-root /mnt/c/Users/ab409/Downloads/PORTER-main/NeMo \
  --model data/lab/gate2/final-attempt-v2/model/final-gate2-selected.nemo \
  --audio data/lab/gate2/final-attempt-v2/test-v2/audio/DKSR20001002.1.1.109.wav \
  --surface-predictions \
    data/lab/gate2/final-attempt-v2/gpu-evaluation/independent-busan/fine-tuned/predictions.jsonl \
  --output-dir data/lab/gate3/smoke-reproduction
```

The command intentionally validates the model and audio hashes. The sample is used only as
an integration smoke; its prior Gate 2 Test status prevents any new generalization claim or
threshold selection from this result.

## Reproduce the frozen engineering batch

Build the transcript-free manifest deterministically, then run from the frozen WSL environment:

```bash
cd /mnt/c/Users/ab409/orca/workspaces/ai-apps/wreckfish
~/.local/bin/uv run python scripts/build_gate3_engineering_benchmark.py \
  --source-manifest data/lab/gate2/final-attempt-v2/validation/manifest.absolute.jsonl \
  --output-dir data/lab/gate3/engineering-benchmark-v3 \
  --benchmark-id gate3-streaming-engineering-v3 \
  --seed gate3-engineering-v3-2026-08-12 \
  --exclude-manifest data/lab/gate3/engineering-benchmark-v1/manifest.jsonl \
  --exclude-manifest data/lab/gate3/engineering-benchmark-v2/manifest.jsonl

/home/ab409/.venvs/task-005-nemo/bin/python \
  external_inference/nemotron_3_5/run_gate3_batch.py \
  --nemo-root /mnt/c/Users/ab409/Downloads/PORTER-main/NeMo \
  --model data/lab/gate2/final-attempt-v2/model/final-gate2-selected.nemo \
  --manifest data/lab/gate3/engineering-benchmark-v3/manifest.jsonl \
  --criteria artifacts/gate-3/status/gate3-criteria.v3.json \
  --end-of-stream-padding-ms 320 \
  --output-dir data/lab/gate3/engineering-v3-reproduction
```

The frozen official v3 run already exists and must not be replaced. A reproduction is expected
to exit 1 when raw exact agreement misses the threshold; do not reinterpret that exit as an
infrastructure error. Inspect `assessment.json`.
