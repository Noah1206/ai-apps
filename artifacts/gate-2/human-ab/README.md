# TASK-005 blinded A/B review

All 10 frozen Benchmark items are ready. The queue contains WAV, reference, dialect labels,
and anonymous candidates A/B; it contains no model identifier. Do not open
`artifacts/task-005/evaluation-v0/output/blinded-ab-review-key.json` until all decisions are
complete.

The existing comparison page reveals model names and does not store every required Gate 2
field, so it is not reused for this blind review. Run this resumable terminal review instead:

```bash
python3 ~/.codex/skills/use-busan-project-venv/scripts/run.py \
  python -m busan_lab.gate2 review-ab \
  --queue artifacts/gate-2/human-ab/blinded-review-queue-v2.json \
  --results artifacts/gate-2/human-ab/blinded-review-results.json \
  --reviewer-id reviewer-01 \
  --open-audio
```

For each item choose `a`, `b`, `s` (same), or `u` (uncertain). Meaning distortion and
overcorrection also accept `d` (both) and `n` (neither). Progress is saved after every item
in the existing Gate 2 result schema. `--open-audio` plays each WAV directly with macOS
`afplay`; it does not launch Apple Music or convert the source file. The command never opens
the model key.

The stored fields are:

- `transcript_preference`: `A`, `B`, `tie`, or `uncertain`;
- `dialect_preservation_preference`: same choices;
- `meaning_fidelity_preference`: same choices;
- `meaning_distortion_a` and `_b`: `present`, `absent`, or `uncertain`;
- `overcorrection_a` and `_b`: same choices;
- optional `notes`.

Set a reviewer pseudonym, then set `status` to `complete`. Validate before opening the key:

```bash
python3 ~/.codex/skills/use-busan-project-venv/scripts/run.py \
  python -m busan_lab.gate2 validate-ab \
  --queue artifacts/gate-2/human-ab/blinded-review-queue-v2.json \
  --results artifacts/gate-2/human-ab/blinded-review-results.json
```

Only after that succeeds, run again with:

```text
--key artifacts/task-005/evaluation-v0/output/blinded-ab-review-key.json
```

The second command reports the anonymous transcript preference mapping. It does not change
the historical Audio Lab reviews or predictions.
