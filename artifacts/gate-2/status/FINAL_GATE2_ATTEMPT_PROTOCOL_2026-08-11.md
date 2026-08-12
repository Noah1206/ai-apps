# Final Gate 2 attempt protocol — 2026-08-11

This document freezes the last authorized Gate 2 improvement attempt before any new model
is trained or any new final-Test prediction is produced.

## One-attempt rule

- Run exactly one new adapter training job.
- Do not change the proposed Gate 2 thresholds.
- Do not use the consumed Independent Busan Test v1 for training, validation, checkpoint
  selection, threshold selection, or final reassessment.
- Select exactly one checkpoint using development Validation loss only; use the minimum
  finite `val_loss`, breaking an exact tie in favor of the earlier checkpoint.
- Evaluate the selected candidate exactly once on the frozen, previously unseen Independent
  Busan Test v2 and on the unchanged Standard Korean Regression v1.
- Record the resulting PASS, CONDITIONAL PASS, or FAIL as final; do not tune and retry.

## Frozen data allocation

All new AI Hub items must have `current_residence=부산`, a source-provided `dialect_form`,
and a source-provided dialect annotation. Items must pass the existing clean-transcript and
2–12 second duration filters.

| Split | New speakers | New utterances | Purpose |
|---|---:|---:|---|
| Train expansion | 40 | 800 (20 each) | Adapter optimization only |
| Checkpoint Validation expansion | 10 | 100 (10 each) | `val_loss` checkpoint selection only |
| Independent Busan Test v2 | 10 | 100 (10 each) | One final Gate 2 evaluation only |

The historical 200-utterance Train pool remains in Train. The historical 40-utterance
checkpoint Validation remains in Validation. Therefore the intended runtime manifests are
1,000 Train utterances and 140 Validation utterances.

Speakers and source recordings must be disjoint across all three new allocations. Every new
allocation must also exclude all historical Train, Validation, benchmark, and consumed Test
v1 speakers, source recordings, utterances, audio lineage hashes, and normalized surface
texts. Test v2 must be written and hashed before training begins, then remain unopened by the
training and checkpoint-selection paths.

## Frozen training configuration

- Base model: `nvidia/nemotron-3.5-asr-streaming-0.6b`
- Model revision: `f3d333391852ba876df169dcc9ba902d25b6ab0b`
- NeMo revision/environment: recovered TASK-005 Python 3.13.14 environment
- Adapter: encoder linear adapter, dimension 32, dropout 0.0
- Optimizer: AdamW, learning rate `5e-4`, weight decay `0.0`
- Scheduler: cosine annealing, 20 warm-up steps, minimum learning rate `1e-5`
- Precision: 16-bit mixed precision
- Batch duration: 8 seconds; gradient accumulation: 8
- Maximum steps: 600; validation interval: 25 steps
- Checkpoints retained: best two by minimum `val_loss`
- Random seed: 0
- Trainable-parameter and non-finite-loss guards remain enabled

The run may be restarted only to recover from an infrastructure failure before a usable
checkpoint exists, without changing data, seed, configuration, or model code. Such recovery
does not authorize another experiment.

## Final decision

The unchanged proposed Gate 2 criteria apply. In particular, Test v2 requires at least 10%
relative CER improvement, at least +0.15 absolute dialect-preservation improvement, and no
additional empty outputs. Standard Korean regression limits and the completed blinded human
A/B evidence remain unchanged.
