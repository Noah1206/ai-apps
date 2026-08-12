# Gate 3 runtime-v3 assessment — superseded 2026-08-12

This assessment preserves the final v1/runtime-v3 result. The current final decision is the
later, disjoint runtime-v4 assessment in `FINAL_GATE3_RUNTIME_V4_ASSESSMENT_2026-08-12.md`.

## Decision

**FAIL.** Runtime v3 passed 28 of 29 thresholds frozen before batch output inspection. The sole
failure is raw exact final agreement with offline Surface ASR: 18/20 (`0.90`) against a minimum
of `0.95`.

No threshold was relaxed. No offline transcript was copied into a streaming result. After the
declared final run 003, no fourth batch was executed.

## Frozen identities

- Model SHA-256:
  `eb7d5112329504db72ffcb2c638e245e68f35801cda14a33682478d4a2ae85ee`
- Engineering manifest SHA-256:
  `09b1685eaeb512ad1b9d5632ba40e6b829387bd5cf3ee0724038da2202d73b7a`
- Criteria SHA-256:
  `bf8831bd69f2f3faed1dce0ebb5c28d42819120c810753a8db76214f41ea8757`
- Runtime v3 streaming runner SHA-256:
  `a556ff1a84ad18c22b1eff4a60dce1dae59901497e8312fa40012e9e7149c5a3`
- Batch runner SHA-256:
  `df6e0016bfb04b6ab43168f8335097ddf8ad0d279cc8385a3d3dfc3ed23f27ca`
- Runtime: Python 3.13.14, NeMo `3.1.0+6c57e73e83`, Torch `2.12.0+cu132`,
  NVIDIA GeForce RTX 2070

The 20 cases cover 10 speakers with two files per speaker and durations from 2.5 to 7.0 seconds.
They were deterministically selected from checkpoint Validation for streaming engineering only.
They support no new ASR quality-generalization claim.

## Final run 003

| Check | Observed | Frozen requirement | Result |
|---|---:|---:|---|
| Complete traces | 20/20 | 20/20 | pass |
| Empty final transcripts | 0 | 0 | pass |
| Non-empty partial coverage | 1.0 | ≥ 1.0 | pass |
| Mean partial stability | 1.0 | ≥ 0.98 | pass |
| Trace retraction rate | 0.0 | ≤ 0.05 | pass |
| Exact Surface agreement | 0.90 | ≥ 0.95 | **fail** |
| Aggregate Surface CER | 0.0 | ≤ 0.01 | pass |
| Warm first-partial p95 | 1376.92ms | ≤ 2500ms | pass |
| Finalization-lag p95 | 141.19ms | ≤ 500ms | pass |
| Chunk inference p95 | 133.54ms | ≤ 320ms | pass |
| Trace real-time-factor p95 | 1.0300 | ≤ 1.15 | pass |
| Synthetic endpoint F1 | 0.95 | ≥ 0.90 | pass |
| Endpoint early-trigger rate | 0.05 | ≤ 0.10 | pass |
| Endpoint delay p95 | 723.0ms | ≤ 1000ms | pass |
| Cancellation / reset | 1.0 / 1.0 | 1.0 / 1.0 | pass |
| Allocated / reserved growth | 0 / 142,606,336 bytes | ≤ 64MiB / 256MiB | pass |
| Adapter execution | 24/24, every session | 24/24 | pass |
| Session state release | every session | every session | pass |
| Confidence policy | explicitly unsupported | same | pass |

The two exact mismatches are audio hashes
`628a3e63b8ed7379a813aa49f0feae353376d2e134dd20b41ed6ef4797102550` and
`64b4c1268d95f52a8b8bf447d7ca67807215181fafa75d8367d157b96871f625`.
In both, streaming omitted only terminal `U+002E` while every CER-counted character matched.

The synthetic endpoint failure is audio hash
`6947382347b6c425e6b328771e5b6e791f03666bfc4d3f93bc9f0a1b50ffb8b9`: the energy baseline
triggered at 1,700ms, 3,750ms before the 5,450ms source boundary. This remained within the
precommitted aggregate allowance but is a required natural-endpoint follow-up.

## Candidate history

- Run 001 / runtime v1: FAIL. Five correct accumulated partials were not published as final when
  NeMo stopped on an insufficient residual feature frame.
- Run 002 / runtime v2: FAIL. Explicit iterator-exhaustion finalization fixed 20/20 completeness;
  two terminal periods remained absent.
- Run 003 / runtime v3: FAIL. Predicting the last yieldable chunk and setting
  `keep_all_outputs=True` preserved all CER-counted text but did not recover the two periods.

## Next experiment

The smallest legitimate next candidate is an explicit end-of-stream cache flush with padded
silence/features and no access to offline text. Before running it, create a new protocol version,
freeze its source/config hashes, and retain the same raw-exact punctuation requirement. Gate 4
does not begin until that candidate passes or the project explicitly accepts a changed product
policy in a separately justified Gate definition.

Local raw run 003 artifact hashes are recorded in the redacted aggregate summary. Licensed
transcripts and per-event traces remain under ignored `data/lab/gate3/engineering-v1-run-003`.
