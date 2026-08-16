# TTS Model Decision

## Decision

- Status: `PREFERRED_PENDING_SMOKE`
- Candidate: `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
- Decision date: 2026-08-16
- Release scope: one consented Busan native speaker, one stable style, offline generation,
  24 kHz WAV cache for 30-50 core lesson sentences

This is the only TTS candidate authorized for the first smoke. Do not start a broad model
comparison. It is not a Release Candidate until the exact weights load on RTX 2070 8 GB,
one consented Busan reference voice generates Korean surface text, and output validation
passes.

## Why This Candidate

- The official project lists Korean among ten supported languages.
- The official 0.6B Base description supports a short reference-audio voice clone and
  fine-tuning, allowing a single stable speaker without adding multi-speaker scope.
- The official code repository is Apache-2.0 and the Hugging Face model card also declares
  Apache-2.0 for the weights.
- The published weight repository is about 2.52 GB, materially smaller than the other
  currently considered multilingual packages and more plausible for the available 8 GB
  GPU. Actual VRAM fit remains UNVERIFIED.
- Zero-shot speaker conditioning can provide the August fallback if a clean, explicitly
  consented reference is available; a larger fine-tuning effort is not required before the
  first product smoke.

## Required Evidence Before Download/Execution

1. A Busan native speaker's explicit consent for reference conditioning, generated voice
   use, app distribution, and storage.
2. One clean 3-10 second reference clip plus an exact Busan surface transcript for the
   first smoke.
3. A decision on whether this voice may be cached in the shipped application.
4. At least 50 held-out target sentences for automated checks; the prompt text may be
   prepared now, but quality claims require generated audio.

## First Smoke Contract

- Load `Qwen/Qwen3-TTS-12Hz-0.6B-Base` in a separate TTS environment.
- Pin repository and model revisions before downloading.
- Generate one short Korean sentence containing a real Busan surface form.
- Preserve input text; do not normalize Busan endings into Standard Korean.
- Save WAV, sample rate, duration, peak, clipping, silence ratio, latency, GPU peak memory,
  model revision, code commit, weight hashes, and reference-audio consent identifier.
- If the model does not fit RTX 2070 8 GB or fails Korean content consistency, record FAIL
  and stop. Do not initiate a model sweep without a new evidence-based decision.

## Prohibited Fallback Claims

The built-in Korean `Sohee` voice in the CustomVoice variant may be useful for an API
plumbing smoke, but there is no evidence it is a Busan native voice. It must not be shipped
or described as Busan TTS solely because it can read Busan text.

## License Record

- Code: Apache License 2.0, official repository
  <https://github.com/QwenLM/Qwen3-TTS>
- Weights: Apache-2.0 metadata, official model repository
  <https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base>
- Speaker rights: BLOCKED; no consent record exists locally

Apache-2.0 metadata does not grant rights to clone an unconsenting person's voice. Speaker
permission remains a separate release gate.
