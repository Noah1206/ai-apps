# Gate 3 evaluation protocol — frozen before batch inference

Status: **FROZEN 2026-08-12 before batch inference**. The implementation candidate,
benchmark identity, and thresholds below cannot be changed after results are opened.

## Fixed implementation candidate

- Gate 2 selected model SHA-256:
  `eb7d5112329504db72ffcb2c638e245e68f35801cda14a33682478d4a2ae85ee`
- NeMo revision: `3.1.0+6c57e73e83`
- Python / Torch: `3.13.14` / `2.12.0+cu132`
- Runtime: `CacheAwareStreamingAudioBuffer` + `conformer_stream_step`
- Decoder: RNNT greedy batch
- Prompt: `ko-KR`, language tags stripped
- Attention context: `[56, 3]`
- Audio: 16kHz mono uncompressed PCM16
- Compute: float32, no AMP
- Stable-prefix window: 3 observations

These values may not change after the benchmark outputs are observed. A changed value is a
new candidate and requires a new protocol version.

### Runtime candidate history

- `runtime-v1`, batch run `engineering-v1-run-001`: failed because final publication depended
  on NeMo `is_buffer_empty()`. Five iterators stopped on an insufficient residual feature frame
  after already producing a correct accumulated hypothesis, so those sessions were never marked
  final.
- `runtime-v2`, declared before `engineering-v1-run-002`: on iterator exhaustion, publish the
  last accumulated hypothesis as the explicit end-of-input final event, matching the lifecycle
  used by NVIDIA's reference `perform_streaming` implementation. Model, decoder, prompt, cache,
  attention context, benchmark, and every numeric threshold remain unchanged. Frozen source
  SHA-256 values are `8566485e4cf35274c7e0d882975520bfef80af9d5442bac2ce6dfd86246b1e04`
  for `run_gate3_streaming.py` and
  `df6e0016bfb04b6ab43168f8335097ddf8ad0d279cc8385a3d3dfc3ed23f27ca`
  for `run_gate3_batch.py`.

Run 001 remains immutable failure evidence. Run 002 is a new runtime candidate evaluated against
the same frozen criteria; it is not an infrastructure retry or threshold adjustment.

- `runtime-v3`, declared before `engineering-v1-run-003`: run 002 completed 20/20 finals and
  achieved aggregate CER 0.0, but two exact comparisons omitted only the final period. The same
  residual-frame boundary had passed `keep_all_outputs=False` to the last yieldable chunk.
  Runtime v3 mirrors NeMo's iterator stop rule before inference and enables `keep_all_outputs`
  on that chunk. It never reads or substitutes the offline transcript. Frozen source SHA-256 is
  `a556ff1a84ad18c22b1eff4a60dce1dae59901497e8312fa40012e9e7149c5a3` for the streaming
  runner and remains
  `df6e0016bfb04b6ab43168f8335097ddf8ad0d279cc8385a3d3dfc3ed23f27ca` for the batch
  runner. All model, input, and threshold values remain unchanged.

## Required evaluation sets

1. `gate3-streaming-engineering-v1`: 20 validation-derived cases, 10 speakers, two cases per
   speaker, 2.5–7.0 seconds. Its transcript-free local manifest SHA-256 is
   `09b1685eaeb512ad1b9d5632ba40e6b829387bd5cf3ee0724038da2202d73b7a`.
2. The same 20 source clips with 1,200ms zero-silence appended for a deterministic synthetic
   endpoint probe. The source-file boundary is the expected end; a detection from 200ms before
   through 1,000ms after that boundary is correct.
3. A runtime lifecycle probe that cancels one session after two chunks and resets/reuses one
   completed session with the same audio.
4. The 20 sequential sessions are also the repeated-session memory probe.

The benchmark was selected before streaming outputs with seed
`gate3-engineering-v1-2026-08-12`: duration filter, SHA-256-ranked eligible speakers, then
SHA-256-ranked utterances. Source manifest SHA-256 is
`6e5c23057e90b84c47141c976138b0faa6a9505342ed09cb70b4cd4fe45fe93c`.

Previously consumed Gate 2 Test v1/v2 may be used only for integration regression. They may
not select thresholds, checkpoints, or ASR parameters and do not provide a new independent
quality claim.

## Metrics to report

- Complete, missing, duplicate, and empty traces
- Partial stability and stable-prefix retraction count
- Character agreement with the frozen Surface ASR final transcript
- Cold and warm first-partial latency under real-time-paced input
- End-of-speech to final transcript latency
- Per-chunk p50/p95 inference latency and real-time factor
- Endpoint precision, recall, and F1
- Cancellation and reset correctness
- Peak memory and memory growth over repeated sessions
- Confidence coverage/calibration, or an explicit unsupported result
- Adapter module/call audit and exact model/config hashes

## Gate decision policy

The machine-readable criteria are frozen in `gate3-criteria.v1.json`. In summary:

- 20/20 complete unique traces, 10/10 speakers, and zero empty final transcripts
- mean partial stability at least `0.98`; no more than 5% of traces with a stable-prefix
  retraction
- at least 95% exact final agreement with offline Surface ASR and aggregate CER at most `0.01`
- real-time-paced warm first-partial p95 at most `2500ms`; finalization lag p95 at most `500ms`;
  post-warmup chunk inference p95 at most `320ms`; trace real-time-factor p95 at most `1.15`
- synthetic endpoint F1 at least `0.90`, early false-trigger rate at most `0.10`, and endpoint
  delay p95 at most `1000ms`
- runtime cancellation and reset/reuse correctness both `1.0`
- allocated-memory growth at most 64MiB and reserved-memory growth at most 256MiB across the
  20 sequential sessions
- all 24 encoder adapters found and called; final buffer state released for every session
- streaming confidence is an explicit unsupported capability for this candidate; no fabricated
  confidence values are permitted

Any missing endpoint, cancellation, reset, confidence-policy, or memory result is a Gate blocker
even if transcription quality is high. Infrastructure failures are rerunnable without changing
the candidate; model/config changes create a new candidate.

The endpoint set is synthetic, not human-annotated natural conversational endpoint data. Passing
it permits this file-based Gate 3 engineering decision but remains a production limitation for
Gate 14 WebRTC work.
