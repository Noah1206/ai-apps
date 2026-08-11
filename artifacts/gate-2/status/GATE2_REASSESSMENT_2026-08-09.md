# Gate 2 reassessment — 2026-08-09

## Gate 2 Status

**FAIL — required generalization evidence is incomplete.**

This status does not reject the adapter. Integrity, reproducibility and the 10-item automatic
pilot pass their proposed checks. Gate 3 must not start while Human A/B and both held-out
evaluations are missing.

## 해결한 문제

- Canonical Benchmark/provenance audit remains unchanged and passing.
- The GPU slim recovery archive was imported without overwriting existing artifacts.
- Archive safety passed; all 125 indexed files match size and SHA-256.
- All 18 required reproducibility artifacts pass their pinned SHA-256 checks.
- Best checkpoint, exported best `.nemo`, embedded adapter state, configuration, launcher,
  patched training entry point, inference runner, environment, locks and logs are preserved.
- A resumable model-blind 10-item terminal review writes the existing Gate 2 JSON contract.
- Held-out dataset validation, paired prediction comparison and one-command Mac assessment
  runner are connected.
- Repository and Downloads candidates were audited against the exclusion registry.

## 아직 남은 문제

- Human A/B: 0/10 reviewed.
- Independent Busan Test: 0 eligible items; proposed minimum 100 / 5 speakers.
- Standard Korean regression: 0 eligible items; proposed minimum 100 / 5 speakers.
- The redundant base pretrained `.nemo` is not in the slim Mac bundle. Its provider revision
  and expected SHA are pinned; the selected full adapter `.nemo` is present.
- No GPU inference was run on the missing held-out datasets.

## Benchmark

Canonical version: `busan-surface-v0@1.0.0`  
Package SHA-256: `151c1e28804627bea69bbd7f6632f4d3558ebf076147e42c1d168d508467233c`  
Semantic manifest SHA-256: `700d352edb4a4e9321b48ec6cd312bec6ad1d4c48fa2bedbcf80a2ca23a67f8c`  
Samples: 10  
Dialect expressions: 15 candidate labels

## Model

Base model: `nvidia/nemotron-3.5-asr-streaming-0.6b`  
Revision: `f3d333391852ba876df169dcc9ba902d25b6ab0b`  
NeMo revision: `6c57e73e83de967eed4d334c493ac313b9afd147`  
Adapter: encoder LinearAdapter, dim 32, 24 modules, 1,622,016 parameters  
Best checkpoint: epoch 5 / global step 97 / val_loss 42.027042  
Checkpoint SHA-256: `a05b3fcc919d18638d87f69700e1e7fba9b2f5f8b2d20396226b2aae9d13343e`  
Exported best `.nemo` SHA-256: `580124ff0ea5c9e2f5546e9186c93c3c2d9e16641a749d02b219e2d06029f950`  
Artifact location: `artifacts/gate-2/reproducibility/recovered/task-005-recovery-gate2-slim-20260808/`

## Evaluation

Busan pilot CER: pretrained `0.6167` → adapter `0.0500`  
Dialect preservation: pretrained `0.2000` → adapter `0.9333`  
Empty outputs: pretrained `3` → adapter `0`  
Human A/B: `0/10`, pending  
Independent speaker test: missing  
Standard Korean regression: missing

The pilot contains only ten items and cannot establish multi-speaker generalization.

## Reproducibility

Artifact-level verification: **PASS**  
Imported bundle: 125/125 files verified  
Required pinned artifacts: 18/18 verified  
Re-execution on this Mac: unavailable because there is no NVIDIA runtime  
External GPU re-execution contract: complete, subject to restoring the pinned base model
artifact and matching its recorded SHA-256.

## Next Gate

Gate 3 is not authorized. Finish, in order:

1. complete the 10-item blinded Human A/B;
2. acquire and freeze an independent Busan Test (at least 100 / 5 new speakers);
3. acquire and freeze a Standard Korean regression set (at least 100 / 5 speakers);
4. run both pinned model arms on the GPU and execute `evaluate-suite` on the Mac;
5. start Gate 3 Streaming ASR integration only if the resulting Gate 2 status is PASS.
