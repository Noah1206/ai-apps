# Gate 2 closure audit

Current status: **FAIL (evidence incomplete)**. This does not invalidate the adapter; it
means Gate 2 cannot be closed yet.

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
- The 10-item blinded A/B queue is ready, but no human decision has been recorded.
- The Train and checkpoint-selection Validation ZIPs are checksum-preserved locally.
- The slim GPU recovery bundle is imported. All 125 indexed files and every required
  reproducibility artifact pass SHA-256 verification.

Blocking evidence:

- Human A/B is incomplete.
- No eligible independent multi-speaker final Test exists.
- No Standard Korean regression set or result exists.

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

No TTS, IPA, new ASR model, model inference, or training was performed during this audit.
