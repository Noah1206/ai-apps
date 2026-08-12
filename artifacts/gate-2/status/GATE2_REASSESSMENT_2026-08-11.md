# Gate 2 reassessment — 2026-08-11

## Decision

**CONDITIONAL PASS**. Evidence collection is complete and 12 checks pass. The only failed
check is independent Busan dialect preservation.

The proposed thresholds were not changed after observing the results.

## Held-out datasets

| Dataset | Source | Utterances | Speakers | Duration | Validation |
|---|---|---:|---:|---:|---|
| Independent Busan Test v1 | AI Hub Korean dialect speech (Gyeongsang), source Validation split | 100 | 5 | 461.690 s | Passed; no errors or warnings |
| Standard Korean Regression v1 | Zeroth-Korean/OpenSLR SLR40 test split | 100 | 10 | 939.013 s | Passed; no errors or warnings |

Both manifests are frozen, disallow training and checkpoint selection, and pass the
Train/Validation/benchmark exclusion registry. AI Hub audio, labels, manifests, and
per-utterance comparisons remain local and are not committed or redistributed. The
Zeroth-Korean source is CC BY 4.0 and is attributed to the Zeroth Project and OpenSLR.

Sources:

- AI Hub dataset 119: <https://aihub.or.kr/aihubdata/data/view.do?aihubDataSe=realm&dataSetSn=119>
- AI Hub usage policy: <https://aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105>
- OpenSLR SLR40: <https://www.openslr.org/40/>
- Parquet conversion used locally: <https://huggingface.co/datasets/kresnik/zeroth_korean>

## GPU evaluation

All four arms completed 100/100 utterances on an NVIDIA GeForce RTX 2070 8 GB using the
recovered Python 3.13.14 environment, PyTorch 2.12.0+cu132, NeMo 3.1.0, the checksum-pinned
pretrained model, and the checksum-pinned TASK-005 adapter. The fine-tuned runs verified
the adapter-state SHA-256 and recorded 2,400 adapter forward calls per dataset.

| Metric | Pretrained | Fine-tuned | Outcome |
|---|---:|---:|---|
| Independent Busan CER | 0.308749 | 0.258140 | 16.39% relative improvement; pass (minimum 10%) |
| Independent dialect preservation | 0.132812 | 0.117188 | delta -0.015625; fail (minimum +0.15) |
| Independent empty outputs | 0 | 0 | pass |
| Standard Korean CER | 0.197736 | 0.141618 | 28.38% relative improvement; no regression |
| Standard empty outputs | 0 | 0 | pass |
| Blinded human preference | — | 8/10 | pass (minimum 70%) |

For the independent set, 39 utterances improved in CER, 33 were equal, and 28 worsened.
The fine-tuned CER improved for four of five speakers on a macro per-speaker view, but exact
dialect-expression preservation did not improve for any speaker.

## Next action

Do not tune the threshold or select a checkpoint against this consumed final Test. Expand
Train and checkpoint-selection Validation with broader, human-reviewed Busan morphology;
train a new fixed candidate; then evaluate it once on a newly frozen, previously unseen
multi-speaker Busan final Test. Keep the current v1 dataset and raw predictions unchanged as
the audit record.
