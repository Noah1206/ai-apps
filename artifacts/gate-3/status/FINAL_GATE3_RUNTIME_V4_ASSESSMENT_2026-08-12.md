# Gate 3 runtime-v4 assessment — superseded 2026-08-12

This assessment preserves the v2/runtime-v4 result. The current final decision is the later,
disjoint runtime-v5 assessment in `FINAL_GATE3_RUNTIME_V5_ASSESSMENT_2026-08-12.md`.

## Decision

**FAIL — 28 of 29 frozen checks passed.** Runtime v4 fixed the v3 terminal-output blocker on a
new, disjoint 20-case/10-speaker benchmark: raw exact agreement reached the required 19/20
(`0.95`). The sole failure is finalization-lag p95, observed at `510.23ms` against the frozen
maximum of `500ms`.

The overrun is `10.23ms`. The threshold was not changed, the run was not repeated, and Gate 4
remains not started.

## Frozen identities

- Runtime candidate: `runtime-v4`
- Model SHA-256:
  `eb7d5112329504db72ffcb2c638e245e68f35801cda14a33682478d4a2ae85ee`
- Engineering v2 manifest SHA-256:
  `509bbdd97e73add605a51355ddaa572365acd1ab4134235bd24d5a1b338c37f3`
- Criteria SHA-256:
  `93a2135c43c7acff0cf525a3ff1c8bd9e1c07eaaad9d585c8fa9473188f8a5fd`
- Streaming runner SHA-256:
  `5afd517aebe556306a76428cc231838fa6910157af84efcec5ec534114365e9b`
- Batch runner SHA-256:
  `5e35ef4fad5ddb331544261415b663efa516bf09c91ac2823bcf9b5dbfb1295f`
- Runtime: Python 3.13.14, NeMo `3.1.0+6c57e73e83`, Torch `2.12.0+cu132`,
  NVIDIA GeForce RTX 2070

The v2 benchmark has no overlapping utterance ID or audio hash with v1. It was deterministically
selected from checkpoint Validation before inference, so it is engineering evidence only and
does not support a new quality-generalization claim.

## Fixed batch result

| Check | Observed | Frozen requirement | Result |
|---|---:|---:|---|
| Complete traces | 20/20 | 20/20 | pass |
| Empty final transcripts | 0 | 0 | pass |
| Non-empty partial coverage | 1.0 | ≥ 1.0 | pass |
| Mean partial stability | 1.0 | ≥ 0.98 | pass |
| Trace retraction rate | 0.0 | ≤ 0.05 | pass |
| Exact Surface agreement | 0.95 | ≥ 0.95 | pass |
| Aggregate Surface CER | 0.0 | ≤ 0.01 | pass |
| Warm first-partial p95 | 1410.58ms | ≤ 2500ms | pass |
| Finalization-lag p95 | 510.23ms | ≤ 500ms | **fail** |
| Chunk inference p95 | 157.23ms | ≤ 320ms | pass |
| Trace real-time-factor p95 | 1.1450 | ≤ 1.15 | pass |
| Synthetic endpoint F1 | 0.90 | ≥ 0.90 | pass |
| Endpoint early-trigger rate | 0.10 | ≤ 0.10 | pass |
| Endpoint delay p95 | 693.0ms | ≤ 1000ms | pass |
| Cancellation / reset | 1.0 / 1.0 | 1.0 / 1.0 | pass |
| Allocated / reserved growth | 0 / 106,954,752 bytes | ≤ 64MiB / 256MiB | pass |
| Adapter execution | 24/24, every session | 24/24 | pass |
| Session state release | every session | every session | pass |
| Confidence policy | explicitly unsupported | same | pass |

The largest finalization lag was `538.52ms` on case `gate3-019`, audio SHA-256
`2d22ef0dace8afbbae8cb12ce4368b81d070aacc0158110bf6f9cd076497f4a5`.
With linear p95 interpolation, it and the second-largest `508.74ms` case produce `510.23ms`.

One raw exact mismatch remains on case `gate3-010`, audio SHA-256
`ab7d9dec824510e105a98398ac8972e6f821baef5271bc230205a8dd1a14594e`: streaming omitted only
terminal `U+003F`. This is within the frozen exact-agreement allowance, and aggregate Surface CER
is 0.0.

## What runtime-v4 proved

The 320ms streaming-only zero-PCM cache flush improves the independent exact-match result from
v3's 18/20 to 19/20 without consulting offline text. It does so at a measurable latency cost:
the flush plus final inference exceeded the 500ms p95 budget by 10.23ms. This is a legitimate
candidate failure, not an infrastructure failure.

The local raw output is `data/lab/gate3/engineering-v2-run-001`; its transcript-redacted summary
and all raw artifact hashes are tracked under `evaluation-results/engineering-v2/summary.json`.
No additional run is allowed under the v2 protocol.
