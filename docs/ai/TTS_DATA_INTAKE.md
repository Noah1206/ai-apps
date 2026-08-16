# TTS Data Intake Gate

TTS training must not start until every required item below is backed by a file or signed
record. Candidate prompts are not training data.

## Required Before Preprocessing

- One Busan native speaker ID and contact/record custodian
- Explicit consent for storage, model training, app inference, generated-audio use, and
  commercial distribution as applicable
- Consent policy version and capture date
- Original recording directory with immutable source files
- Recording device, environment, sample rate, channel count, and session metadata
- Surface transcript for each clip that preserves the spoken Busan form
- File-to-sentence ID mapping
- Exclusion list for withdrawn or unusable recordings

## Minimum Release Target

- Recommended clean duration: 30-60 minutes
- Recommended prompts: 200-300
- One speaker and one stable style
- Speaker-disjoint evaluation is not possible for a one-speaker RC; hold out at least 50
  unseen sentences and keep their audio/text out of training decisions
- Core 30-50 lesson sentences must be generated, automatically checked, and manually
  approved before app caching

A shorter consented 3-10 second reference may be used only for the preferred model's
zero-shot smoke. If the August RC uses that path before the 30-60 minute collection is
complete, report the actual data duration and mark speaker naturalness and stability
`UNVERIFIED` until native review; do not imply that full adaptation training occurred.

## Validation Before Training

- Decode every file
- Record SHA-256, duration, sample rate, channels, codec, peak, clipping, silence, and RMS
- Reject empty, corrupt, clipped, mostly silent, and transcript-mismatched recordings
- Detect duplicate audio hashes and near-duplicate prompts
- Freeze Train/Validation/held-out manifests and record their hashes
- Record speaker consent evidence separately from the public model package
- Confirm model code license and weight license independently

## Current Result

`BLOCKED`: no eligible speaker audio, consent record, TTS model decision, or checkpoint was
found in the repository or linked local assets on 2026-08-16. Do not claim TTS readiness.
