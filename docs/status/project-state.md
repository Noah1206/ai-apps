# Project state — 2026-08-12

## Current decision

- Gate 2 Surface ASR Improvement: **PASS (13/13)**.
- Gate 3 Streaming ASR: **FAIL; 28/29 frozen engineering checks passed**.
- Gate 4 and later research gates: not started by this change.

## Completed

- Local FastAPI Audio Lab with upload, byte preservation, deterministic audio derivatives,
  quality/analysis views, utterance/consent schemas, benchmark/evaluation storage, and static UI.
- Surface ASR provider boundary, precomputed predictions, CER/dialect preservation/error export,
  experiment records, and human review/comparison flows.
- Training-data contracts, consent/label/quality gates, split leakage checks, import/review/export.
- Gate 2 frozen adapter training, independent Busan and standard Korean evaluation, blinded A/B,
  and final PASS evidence.
- Gate 3 event/metrics schemas, stable-prefix tracker, lifecycle tests, and reference-path NeMo
  cache-aware runner.
- Runtime-v4 streaming-only end flush and fixed RTX 2070 evaluation.
- Runtime-v5 immediate post-EOF flush pacing, disjoint v3 commitment, and fixed GPU evaluation.

## Partially implemented

- Streaming ASR accepts a WAV file and simulates 320ms cache-aware chunks on one GPU. It is not
  connected to microphone/WebRTC input or the FastAPI UI.
- Session reset/cancellation, buffer release, and sequential-session memory are tested on the GPU.
- Stable prefix/suffix and final Surface agreement are measured. Endpointing evidence is limited
  to appended silence; confidence is explicitly unsupported rather than fabricated.

## Missing for the next decision

- An adaptive terminal-output design that preserves runtime-v5's passing 246.17ms finalization
  p95 while reaching 0.95 raw exact agreement on unseen clips
- A new precommitted candidate and new unseen engineering set; the completed v3 protocol permits
  no retry
- Live microphone/WebRTC media gateway and natural endpoint annotations remain future production
  work; the current endpoint evidence is synthetic appended silence
- Browser microphone recording remains absent; current browser workflow is file upload/playback

## Environment and repository

- Branch: `Noah1206/clone-copy-next-steps`
- Remote: `origin` → `https://github.com/Noah1206/ai-apps.git`
- Git tag: no Audio Lab tag was found; no tag was created from the current dirty worktree.
- Project package manager: `uv`; project Python requirement is `>=3.12`.
- Frozen GPU environment: WSL Python 3.13.14, uv 0.12.2, NeMo 3.1.0, Torch
  2.12.0+cu132, RTX 2070 8GB.
- Host media tools: FFmpeg/FFprobe 9.0.

## Commands

Project checks from WSL:

```bash
cd /mnt/c/Users/ab409/orca/workspaces/ai-apps/wreckfish
~/.local/bin/uv run ruff check .
~/.local/bin/uv run mypy src
~/.local/bin/uv run pytest
```

The exact Gate 3 GPU smoke command is in `artifacts/gate-3/README.md`.

## Verification result

- Ruff lint: passed for `src`, `tests`, and the Gate 3 external runner.
- Mypy: passed for all 36 source files.
- Gate 3 focused tests: 19 passed.
- Full pytest: 98 passed, 2 failed, 1 deprecation warning.
- The two failures are missing-local-asset failures, not assertion regressions:
  - `data/lab/reports/task-002-nvidia-korean-conformer-ctc-pretrained-v0--task-003a-surface-asr-evaluation-v1.json`
  - `data/lab/manifests/busan-surface-v0--1.0.0.json`
- Full repository `ruff format --check` also reports 19 pre-existing files that would be
  reformatted. Gate 3 files pass their focused format check; unrelated dirty files were not
  rewritten.

## Data and model state

- Licensed audio, transcripts, manifests, checkpoints, and raw traces remain ignored under
  `data/lab/`.
- Gate 2 selected `.nemo` SHA-256:
  `eb7d5112329504db72ffcb2c638e245e68f35801cda14a33682478d4a2ae85ee`.
- Gate 2 Test v1/v2 are consumed and cannot select future thresholds, checkpoints, or model
  parameters.
- Gate 3 accepted smoke evidence is transcript-redacted under `artifacts/gate-3/`.

## Current bottleneck and external input

Runtime-v5's immediate EOF flush fixed latency at 246.17ms p95, but exact agreement varied to
0.65 on the third disjoint set. The next candidate needs a transcript-stability-based adaptive
flush or equivalent model-native terminal mechanism, with a frozen maximum compute budget and no
offline-text access. It requires a new protocol and unseen engineering input; the v3 batch cannot
be retried.
