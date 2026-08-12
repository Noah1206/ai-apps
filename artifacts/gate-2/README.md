# Gate 2 closure audit

Current status: **PASS**. The final one-attempt candidate passes all 13 frozen Gate 2
checks on the previously unseen Independent Busan Test v2 and the unchanged Standard
Korean Regression v1.

Verified now:

- `busan-surface-v0@1.0.0` has two byte serializations but one validated semantic
  identity. Both originals remain unchanged.
- Canonical package SHA-256:
  `151c1e28804627bea69bbd7f6632f4d3558ebf076147e42c1d168d508467233c`.
- Canonical semantic manifest SHA-256:
  `700d352edb4a4e9321b48ec6cd312bec6ad1d4c48fa2bedbcf80a2ca23a67f8c`.
- Re-evaluating the unchanged TASK-005 predictions gives the same pilot results:
  pretrained CER `0.6167`, preservation `0.2000`; adapter CER `0.0500`, preservation
  `0.9333`; empty outputs `3 -> 0`.
- The blinded A/B review is complete: the fine-tuned arm was preferred on 8/10 decisive
  items. The key was not opened or recovered until after all decisions were recorded.
- The Train and checkpoint-selection Validation ZIPs are checksum-preserved locally.
- The slim GPU recovery bundle is imported. All 125 indexed files and every required
  reproducibility artifact pass SHA-256 verification.
- The final local-only Independent Busan Test v2 contains 100 utterances from ten
  previously unused AI Hub speakers. It was frozen before training and has no validation
  errors or warnings.
- A new Zeroth-Korean Standard Korean regression set contains 100 utterances from ten
  test-split speakers under CC BY 4.0. It has no exclusion-registry errors or warnings.
- The final candidate was selected only by minimum finite Validation loss, then evaluated
  once on Test v2. The two new Test-v2 arms and the new Standard fine-tuned arm completed
  100/100 on an RTX 2070 under Python 3.13.14; the unchanged Standard baseline prediction
  was reused.
- Independent Busan CER improved `0.3628 -> 0.2093` (relative improvement `42.30%`) and
  dialect preservation improved `0.1815 -> 0.3629` (delta `+0.1815`). Empty outputs
  remained `0 -> 0`.
- Standard Korean CER improved `0.1977 -> 0.0761` (relative improvement `61.51%`) with
  empty outputs `0 -> 0`.
- All 13 checks pass; there are no failed or pending checks.

Key files:

- Benchmark audit: `benchmark-integrity/benchmark-identity-audit.json`
- No-inference re-evaluation: `benchmark-integrity/task-005-canonical-reevaluation.json`
- Reproducibility contract: `reproducibility/task-005-reproducibility-spec.json`
- Recovery verification: `reproducibility/recovery-bundle-verification.json`
- Blinded review: `human-ab/blinded-review-queue-v2.json`
- Leakage exclusions: `evaluation-datasets/exclusion-registry.json`
- Local candidate audit: `evaluation-datasets/candidate-audit-2026-08-09.json`
- Proposed thresholds: `status/gate2-criteria.proposed.json`
- Current assessment: `status/gate2-assessment.current.json`
- Final reassessment: `status/GATE2_FINAL_REASSESSMENT_2026-08-12.md`
- Final-attempt protocol: `status/FINAL_GATE2_ATTEMPT_PROTOCOL_2026-08-11.md`
- Non-sensitive final evaluation summaries: `evaluation-results/v2/`
- Prior consumed evaluation record: `evaluation-results/v1/`

Detailed per-utterance comparisons, licensed source transcripts, audio, checkpoints, and
model weights remain local and are ignored by Git. Gate 2 is closed; later work must not
reuse either consumed Busan final Test for tuning or checkpoint selection.
