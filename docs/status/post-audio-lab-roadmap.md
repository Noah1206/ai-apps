# Post-Audio-Lab roadmap — current slice

The project follows the Gate 0–15 order in `nextplan.md`, while product integration may proceed
in a separate offline vertical-slice track. Current work is Gate 3.

| Stage | State | Next proof required |
|---|---|---|
| Gate 0 Audio Lab | implemented baseline | Tag only from a clean, intentionally reviewed commit |
| Gate 1 Surface ASR baseline | complete | Preserve evidence |
| Gate 2 Surface ASR improvement | PASS | Preserve frozen model and consumed-set rules |
| Gate 3 Streaming ASR | FAIL (28/29) | Adaptive terminal agreement while preserving passing latency |
| Gate 4 Dialect G2P / Target IPA | not started | Begin only after Gate 3 decision |
| Gates 5–15 | not started | Follow dependencies in `nextplan.md` |

## Immediate Gate 3 sequence

1. Preserve runtime-v5 run 001 as FAIL; do not relax the 0.95 exact-agreement threshold.
2. Treat the v3 benchmark as consumed; do not rerun it as official evidence.
3. Design an adaptive immediate-flush stop rule without reading offline text.
4. Freeze that new candidate and a disjoint unseen input set before evaluation.
5. Start Gate 4 only after Gate 3 passes or the project explicitly revises the Gate policy.

The accepted single-file smoke proves feasibility only. The fixed batch is engineering evidence,
not a new ASR quality-generalization claim.
