# Final Gate 3 runtime-v5 assessment — 2026-08-12

## Decision

**FAIL — 28 of 29 frozen checks passed.** Runtime-v5 fixed runtime-v4's latency blocker on a
new benchmark with no audio or utterance overlap with v1/v2. Finalization-lag p95 fell from
`510.23ms` to `246.17ms`, comfortably passing the frozen `500ms` maximum. The sole failure is raw
exact agreement with unpadded offline Surface ASR: 13/20 (`0.65`) against the frozen minimum of
19/20 (`0.95`).

The threshold was not changed, the run was not repeated, and Gate 4 remains not started.

## Frozen identities

- Runtime candidate: `runtime-v5`
- Model SHA-256:
  `eb7d5112329504db72ffcb2c638e245e68f35801cda14a33682478d4a2ae85ee`
- Engineering v3 manifest SHA-256:
  `8bfbd54d0e1b631fc502773472a056b0b77891ae520cb6618c5d329eb2722b9b`
- Criteria SHA-256:
  `431cd7fae1635f9046ef97d0921a832e2f0d5982657713aec3fbaadb37f3f23f`
- Streaming runner SHA-256:
  `8443ba692bd32b0560b1fd25bd5b4c4d1395517c076fe2d8ba53571fc3052771`
- Batch runner SHA-256:
  `5e35ef4fad5ddb331544261415b663efa516bf09c91ac2823bcf9b5dbfb1295f`
- Runtime: Python 3.13.14, NeMo `3.1.0+6c57e73e83`, Torch `2.12.0+cu132`,
  NVIDIA GeForce RTX 2070

The v3 benchmark contains 20 cases from 10 speakers and has zero overlapping utterance IDs or
audio hashes with the combined v1/v2 benchmarks. It was selected before inference from
checkpoint Validation, so it is engineering evidence only and does not support a new ASR
quality-generalization claim.

## Fixed batch result

| Check | Observed | Frozen requirement | Result |
|---|---:|---:|---|
| Complete traces | 20/20 | 20/20 | pass |
| Empty final transcripts | 0 | 0 | pass |
| Non-empty partial coverage | 1.0 | ≥ 1.0 | pass |
| Mean partial stability | 1.0 | ≥ 0.98 | pass |
| Trace retraction rate | 0.0 | ≤ 0.05 | pass |
| Exact Surface agreement | 0.65 | ≥ 0.95 | **fail** |
| Aggregate Surface CER | 0.00383 | ≤ 0.01 | pass |
| Warm first-partial p95 | 1007.97ms | ≤ 2500ms | pass |
| Finalization-lag p95 | 246.17ms | ≤ 500ms | pass |
| Chunk inference p95 | 143.41ms | ≤ 320ms | pass |
| Trace real-time-factor p95 | 1.0688 | ≤ 1.15 | pass |
| Synthetic endpoint F1 | 0.95 | ≥ 0.90 | pass |
| Endpoint early-trigger rate | 0.05 | ≤ 0.10 | pass |
| Endpoint delay p95 | 783.0ms | ≤ 1000ms | pass |
| Cancellation / reset | 1.0 / 1.0 | 1.0 / 1.0 | pass |
| Allocated / reserved growth | 0 / 106,954,752 bytes | ≤ 64MiB / 256MiB | pass |
| Adapter execution | 24/24, every session | 24/24 | pass |
| Session state release | every session | every session | pass |
| Confidence policy | explicitly unsupported | same | pass |

The seven raw mismatches comprise three missing terminal periods, two missing terminal question
marks, one extra terminal period, and one stream missing two terminal CER-counted characters.
Only the last mismatch contributes to aggregate CER, which still passes at `0.00383`.

The largest finalization lag was `246.20ms`; linear p95 is `246.17ms`. This confirms that
processing synthetic flush PCM immediately after explicit EOF removes the runtime-v4 latency
blocker without hiding flush inference time.

## Next candidate boundary

Runtime-v5 shows that a fixed 320ms immediate flush is fast but not sufficiently robust for raw
terminal agreement on unseen clips. The smallest defensible next experiment is an adaptive,
immediate EOF flush with a transcript-stability stop rule and a precommitted maximum number of
zero-PCM steps. It must not inspect offline text, and its compute time must remain in finalization
lag. The v3 benchmark is now consumed; any official runtime-v6 decision requires a new disjoint
input commitment and protocol.

The local raw output is `data/lab/gate3/engineering-v3-run-001`; its transcript-redacted summary
and raw artifact hashes are tracked under `evaluation-results/engineering-v3/summary.json`.
