# Busan Surface ASR Candidate Model Card

## Identity

- Version: `busan-asr-gate2-pass-20260812`
- Base: `nvidia/nemotron-3.5-asr-streaming-0.6b`
- Architecture: prompt-conditioned FastConformer RNNT
- Adaptation: encoder linear adapter `busan_ko_kr_v0`, dimension 32
- Runtime mode: offline per utterance, `target_lang=ko-KR`
- Decoder: `greedy_batch`, `max_symbols=10`
- Checkpoint SHA-256: `eb7d5112329504db72ffcb2c638e245e68f35801cda14a33682478d4a2ae85ee`
- Adapter SHA-256: `d0716aff0f05580d51dcad138c3036128d925d4821149842625f760ed0b7b954`

## Intended Use

Short Korean lesson-practice recordings where preserving written Busan surface forms is
important. The August release path uses raw offline output. It is not a claim of universal
Busan-dialect recognition, streaming support, pronunciation scoring, or open-ended speech
understanding.

## Data and Selection

The final attempt used 1,000 Train utterances from 41 speakers and 140 checkpoint
Validation utterances from 14 disjoint speakers. The candidate was selected by minimum
Validation loss only. Independent Busan Test v2 contained 100 utterances/10 speakers and
Standard Korean Regression v1 contained 100/10. Speaker, utterance, audio hash, and
normalized surface overlap checks were empty across the final splits.

Test v1 and Test v2 are consumed. Neither may be reopened for tuning or another release
selection.

## Evaluation

| Metric | Pretrained | Candidate |
|---|---:|---:|
| Independent Busan CER | 0.362765 | 0.209325 |
| Dialect preservation | 0.181467 | 0.362934 |
| Standard Korean CER | 0.197736 | 0.076108 |
| Empty outputs | 0 | 0 |

Blinded human preference passed at 8/10. Gate 2 passed all 13 frozen checks. These are raw
model results without context biasing, candidate restriction, or text post-processing.

## Reproducibility

- Python 3.13.14
- PyTorch 2.12.0+cu132
- NeMo 3.1.0 at Speech commit `6c57e73e83de967eed4d334c493ac313b9afd147`
- CUDA 13.2
- GPU: NVIDIA RTX 2070 8 GB
- Seed: 0
- Training: AdamW, learning rate `5e-4`, cosine annealing, mixed precision 16,
  accumulation 8, maximum 600 steps
- Selected checkpoint: epoch 2, global step 269, `val_loss=16.316898345947266`

## License

The NVIDIA base model card lists OpenMDW-1.1 as the governing model license. Code and
model-weight licenses are separate. Before public distribution, retain the base notice and
complete a derivative-model/commercial-deployment compliance review. This file does not
replace legal approval.

## Limitations and Risks

- Dialect preservation is 0.362934, not near-perfect.
- Performance is measured on finite Busan and Standard Korean sets and may not generalize
  to unseen microphones, noise, age groups, or other Gyeongsang varieties.
- A 50-file 3-8 second Train stability run completed 50/50 with no failures or empty
  outputs. Mean latency was 563 ms, p50 387 ms, p95 1,239 ms, and mean RTF 0.156 on RTX
  2070. This stability run is not a held-out quality evaluation.
- The model may standardize, omit, insert, or substitute dialect expressions.
- Confidence is not exposed by the frozen inference path.
- API outputs must display the model version and retain raw text when any future product
  post-processing is added.

## Evidence

- `artifacts/gate-2/status/GATE2_FINAL_REASSESSMENT_2026-08-12.md`
- `data/lab/gate2/final-attempt-v2/assessment/gate2-assessment.json`
- `data/lab/gate2/final-attempt-v2/model/export-audit.json`
- `artifacts/release/speech-api/asr-single-file-smoke-20260816.json`
- `artifacts/release/speech-api/http-end-to-end-smoke-20260816.json`
- `artifacts/release/speech-api/asr-50-sequential-stability-20260816.json`
- `artifacts/release/speech-api/verification-summary-20260816.json` (tracked non-sensitive
  aggregate; detailed release smoke files remain local-only)
