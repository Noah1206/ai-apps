# Gate 2 held-out evaluation datasets

No eligible final Test dataset was found on this Mac. The 200-item Train pool, 40-item
checkpoint-selection Validation set, and 10-item frozen pilot are excluded by speaker,
utterance, source-recording, audio-lineage and exact-transcript indexes in
`exclusion-registry.json`.

Two new immutable datasets are required:

1. `independent_busan_test`: actual Busan/Gyeongsang speech with human-reviewed surface
   transcripts and dialect-expression labels;
2. `standard_korean_regression`: human-reviewed Standard Korean speech, with no use for
   training or checkpoint selection.

Both must be 16 kHz mono PCM16 WAV, frozen, licensed for this evaluation, and disjoint from
all exclusions. The proposed minimum is 100 utterances / 5 speakers each; 200 / 10 is
recommended. The minimum is a pipeline decision, not a statistical guarantee: it prevents
another one-speaker test and permits basic per-speaker inspection while keeping the first
Gate 2 audit feasible.

The JSON schema is `../schemas/gate2-evaluation-dataset.schema.json`. Validate a collected
manifest with:

The repository and Downloads audit is recorded in
`candidate-audit-2026-08-09.json`: eligible Busan items `0`, eligible Standard Korean items
`0`. No synthetic substitute was created.

```bash
python3 ~/.codex/skills/use-busan-project-venv/scripts/run.py \
  python -m busan_lab.gate2 validate-dataset \
  --manifest /path/to/dataset/manifest.json \
  --exclusions artifacts/gate-2/evaluation-datasets/exclusion-registry.json \
  --criteria artifacts/gate-2/status/gate2-criteria.proposed.json \
  --output /path/to/dataset/validation-report.json
```

After validation, run the same pinned pretrained and selected adapter once on both datasets.
Use existing Audio Lab normalization for CER; retain raw predictions. Report CER, empty
outputs and latency for both. For the Busan set also report dialect preservation. Do not use
either result to select a checkpoint.

When the two manifests and four GPU prediction files exist, this single Mac command validates
both datasets against the exclusion registry, compares raw predictions, incorporates the
completed blind review, and writes the Gate 2 assessment:

```bash
python3 ~/.codex/skills/use-busan-project-venv/scripts/run.py \
  python -m busan_lab.gate2 evaluate-suite \
  --independent-manifest /path/to/busan/manifest.json \
  --independent-pretrained /path/to/busan/pretrained.jsonl \
  --independent-fine-tuned /path/to/busan/fine-tuned.jsonl \
  --standard-manifest /path/to/standard/manifest.json \
  --standard-pretrained /path/to/standard/pretrained.jsonl \
  --standard-fine-tuned /path/to/standard/fine-tuned.jsonl \
  --exclusions artifacts/gate-2/evaluation-datasets/exclusion-registry.json \
  --criteria artifacts/gate-2/status/gate2-criteria.proposed.json \
  --base-evidence artifacts/gate-2/status/gate2-evidence.current.json \
  --repro-verification artifacts/gate-2/reproducibility/current-verification.json \
  --bundle-verification artifacts/gate-2/reproducibility/recovery-bundle-verification.json \
  --queue artifacts/gate-2/human-ab/blinded-review-queue-v2.json \
  --results artifacts/gate-2/human-ab/blinded-review-results.json \
  --key artifacts/task-005/evaluation-v0/output/blinded-ab-review-key.json \
  --output-dir artifacts/gate-2/evaluation-results/v1
```

GPU inference remains a separate external step; this command does not load NeMo on the Mac.

On the NVIDIA/WSL machine, `run_gate2_nemo_suite.py` reuses the exact checksum-pinned
TASK-005 runner and runs both datasets against both arms in one command:

```bash
python external_inference/nemotron_3_5/run_gate2_nemo_suite.py \
  --recovered-runner artifacts/gate-2/reproducibility/recovered/task-005-recovery-gate2-slim-20260808/inference/run_task005_nemo_benchmark.py \
  --independent-manifest /data/busan-test/manifest.json \
  --standard-manifest /data/standard-test/manifest.json \
  --pretrained-model /models/nemotron-3.5-asr-streaming-0.6b.nemo \
  --pretrained-sha256 210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74 \
  --pretrained-identifier hf://nvidia/nemotron-3.5-asr-streaming-0.6b@f3d333391852ba876df169dcc9ba902d25b6ab0b \
  --fine-tuned-model artifacts/gate-2/reproducibility/recovered/task-005-recovery-gate2-slim-20260808/model/task-005-busan-adapter-best-epoch5-step97.nemo \
  --fine-tuned-sha256 580124ff0ea5c9e2f5546e9186c93c3c2d9e16641a749d02b219e2d06029f950 \
  --adapter-sha256 f2c17a1c2bfdb8cf9f24ad79f28b6c3379c48ce1f3067312bb09e4b0eb41ff36 \
  --fine-tuned-identifier task-005://best-epoch5-step97 \
  --model-revision f3d333391852ba876df169dcc9ba902d25b6ab0b \
  --warmup-audio /data/non-evaluation-warmup.wav \
  --output-dir /output/gate2
```

The wrapper removes reference text before invoking NeMo. It refuses non-frozen or
checkpoint-selectable manifests, mismatched audio hashes, and a modified historical runner.
The four resulting `predictions.jsonl` files are the inputs to the Mac command above.
