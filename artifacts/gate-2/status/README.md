# Proposed Gate 2 decision rule

All numeric values in `gate2-criteria.proposed.json` are **proposed thresholds**, not
retrospectively proven constants.

- Pilot CER <= 0.35 and preservation >= 0.60: requires a substantial result on the current
  pilot but never replaces held-out evidence. The current adapter already clears these.
- Blinded preference >= 0.70 among decisive A/B judgments: requires a clear human tendency
  on the small 10-item pilot.
- Independent Test: at least 100 utterances / 5 speakers, >=10% relative CER improvement,
  >=0.15 absolute preservation improvement, and no additional empty outputs.
- Standard Korean: at least 100 utterances / 5 speakers, CER regression no greater than the
  larger of 0.03 absolute or 15% relative, and no additional empty outputs.

Decision:

- `PASS`: every integrity, reproducibility, human, held-out Busan and Standard Korean check
  passes.
- `CONDITIONAL PASS`: all required evidence exists, all integrity/safety checks pass, but
  exactly one non-safety improvement threshold misses. The miss must be documented before
  proceeding.
- `FAIL`: evidence is missing, integrity/reproducibility fails, or any safety regression
  fails.

The final one-attempt evidence passes every criterion. The current result is `PASS`; see
`GATE2_FINAL_REASSESSMENT_2026-08-12.md` for the frozen datasets, selected checkpoint,
runtime exception disclosure, metrics, and final decision.
