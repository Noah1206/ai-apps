# Gate 2 final reassessment — 2026-08-12

## Decision

**PASS**. The final one-attempt candidate passes all 13 frozen Gate 2 checks. There are no
failed or pending checks. The proposed thresholds were not changed after observing either
the prior consumed Test v1 result or this final Test v2 result.

## Final-attempt data contract

Independent Busan Test v2 was allocated first and hashed before training. The builder then
allocated checkpoint Validation and Train data while excluding all historical Train,
Validation, benchmark, and consumed Test v1 identities. Licensed source audio, labels, and
manifests remain local.

| Split | Utterances | Speakers | Purpose |
|---|---:|---:|---|
| Train | 1,000 | 41 | Adapter optimization |
| Checkpoint Validation | 140 | 14 | `val_loss` selection only |
| Independent Busan Test v2 | 100 | 10 | One final evaluation |
| Standard Korean Regression v1 | 100 | 10 | Unchanged regression evaluation |

Train, Validation, and Test v2 have zero overlap by speaker, utterance, audio identity, and
normalized surface text. Test v2 passed validation without errors or warnings. Its manifest
SHA-256 is
`1e4692d1a0e164756ba4d5537b30733c0aef07e7a7a12ef7e6c2a260a8ea3839`.

## Training and checkpoint selection

The frozen adapter configuration used Python 3.13.14, NeMo 3.1.0, the pinned Nemotron 3.5
ASR base model, seed 0, an encoder linear adapter of dimension 32, AdamW at `5e-4`, cosine
annealing, 16-bit mixed precision, effective gradient accumulation 8, and a maximum of 600
steps.

The initial process was killed by WSL memory pressure before a usable checkpoint existed.
The allowed infrastructure recovery increased WSL memory/swap and restarted the same
data, seed, configuration, and code. After usable checkpoints existed, that single job
stopped at global step 327 with PyTorch GradScaler's
`AssertionError: No inf checks were recorded for this optimizer.` It was not restarted.
The requested 600 steps were a maximum, not a required selection point.

Across 103 finite Validation events, the frozen rule selected the minimum `val_loss`:

- epoch 2, global step 269, `val_loss=16.316898345947266`
- checkpoint SHA-256:
  `2550030143692e147bd73dbae804f9d46ef9ee5c9fd57a85c0cc1e6a6c5a22fb`
- exported `.nemo` SHA-256:
  `eb7d5112329504db72ffcb2c638e245e68f35801cda14a33682478d4a2ae85ee`
- 96 adapter tensors / 1,622,016 adapter parameters; checkpoint and exported adapter-state
  SHA-256 both
  `d0716aff0f05580d51dcad138c3036128d925d4821149842625f760ed0b7b954`
- zero missing or unexpected adapter keys during export

## GPU evaluation

Evaluation used the RTX 2070 under the recovered Python 3.13.14 / NeMo 3.1.0 environment.
The selected candidate was evaluated exactly once on the frozen Busan Test v2 and the
unchanged Standard set. Both Test-v2 arms and the Standard fine-tuned arm completed 100/100
with no failed, missing, or duplicate predictions. Fine-tuned runs recorded 24 registered
adapter modules, 24 warm-up adapter calls, and 2,400 benchmark adapter calls. The unchanged
Standard pretrained prediction was reused from the prior evaluation rather than rerun.

| Metric | Pretrained | Final candidate | Criterion and outcome |
|---|---:|---:|---|
| Independent Busan CER | 0.362765 | 0.209325 | 42.30% relative improvement; pass (minimum 10%) |
| Independent dialect preservation | 0.181467 | 0.362934 | delta +0.181467; pass (minimum +0.15) |
| Independent empty outputs | 0 | 0 | pass |
| Standard Korean CER | 0.197736 | 0.076108 | 61.51% relative improvement; no regression |
| Standard empty outputs | 0 | 0 | pass |
| Blinded human preference | — | 8/10 | pass (minimum 70%) |

On Test v2, 79 utterances improved in CER, 12 were equal, and 9 worsened. On Standard
Korean, 90 improved, 3 were equal, and 7 worsened.

## Final checks

The passing checks are benchmark integrity, reproducibility, all three pilot checks,
blinded human A/B, independent dataset size, independent CER improvement, independent
dialect preservation, independent empty-output regression, standard dataset size,
standard CER regression, and standard empty-output regression.

Tracked non-sensitive evidence:

- final assessment: `gate2-assessment.current.json`
- final evidence: `gate2-evidence.current.json`
- thresholds: `gate2-criteria.proposed.json`
- one-attempt protocol: `FINAL_GATE2_ATTEMPT_PROTOCOL_2026-08-11.md`
- dataset validation, assessment, evidence, and checkpoint selection:
  `../evaluation-results/v2/`

## Next gate

Gate 2 is closed. Gate 3 Streaming ASR integration may begin. Both Busan final Test v1 and
v2 are consumed evaluation sets and must not be reused for training, Validation,
checkpoint selection, or threshold selection.
