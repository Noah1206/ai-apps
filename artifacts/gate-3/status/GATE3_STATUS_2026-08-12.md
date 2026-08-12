# Gate 3 status — 2026-08-12

## Decision

**FAIL — 28/29 frozen checks passed.**

The frozen Gate 2 adapter can run through NeMo's cache-aware FastConformer-RNNT step API on
the RTX 2070. It publishes partial and final transcript events, preserves encoder state
between chunks, and releases the input buffer at session end.

The latest frozen runtime-v5 batch used a third 20-case/10-speaker set with zero audio-hash and
utterance-ID overlap with the combined v1/v2 sets. It processes the 320ms streaming-only zero-PCM
flush immediately after explicit EOF. Finalization-lag p95 fell to 246.17ms and passed the 500ms
budget. It passed every other stability, CER, latency, endpoint, lifecycle, memory, adapter,
state-release, confidence-policy, and infrastructure check. It failed only raw exact Surface
agreement at 13/20 (0.65) against the precommitted 19/20 (0.95) minimum. Criteria were not changed
and the batch was not repeated.

## Accepted smoke

The accepted run is the single-process local `data/lab/gate3/smoke-v3` result. It used the
frozen `.nemo` SHA-256
`eb7d5112329504db72ffcb2c638e245e68f35801cda14a33682478d4a2ae85ee`, Python 3.13.14,
NeMo `3.1.0+6c57e73e83`, Torch `2.12.0+cu132`, and an NVIDIA GeForce RTX 2070.

| Observation | Result |
|---|---:|
| Audio duration | 2,930ms |
| Chunk size / attention context | 320ms / `[56, 3]` |
| Events | 10 total, 9 partial, 1 final |
| First non-empty partial | 1,813.39ms accelerated-run wall clock |
| Final transcript | 2,187.24ms accelerated-run wall clock |
| Mean chunk inference | 217.29ms |
| Partial stability | 1.0, 0 stable-prefix violations |
| Final agreement with Surface ASR | 1.0, CER 0.0 |
| Peak allocated GPU memory | 2,694,814,208 bytes |
| Adapter execution | 24 modules, 240 calls |
| Session buffer release | passed |

The first CUDA step took about 1.50 seconds while later chunks took roughly 49–151ms. This
is a cold, accelerated file simulation, not a real-time paced latency measurement.

## Integration finding

The local NeMo unified `PipelineBuilder` path returned ten empty events for this prompt model.
The official reference cache-aware path using `CacheAwareStreamingAudioBuffer` and
`conformer_stream_step` returned the correct transcript and matched offline output. The
tracked runner therefore uses the validated reference path. It remains isolated under
`external_inference/` so the model-neutral project schema does not depend on NeMo.

## Completed scope

- Strict partial/final event and trace metrics schemas
- Stable prefix / unstable suffix computation
- Explicit active, finalized, cancelled, and reset lifecycle
- Monotonic audio timeline validation
- Cache-aware GPU runner with frozen model/audio hash checks
- `ko-KR` inference prompt and language-tag stripping
- Encoder adapter execution audit
- Surface final agreement joined by audio SHA-256 only
- Unit tests for lifecycle, retraction, schema consistency, audio contract, and runner source
- Streaming-only 320ms zero-PCM end-of-stream cache flush, with unpadded offline comparison
- Three mutually input-disjoint benchmark commitments and fixed RTX 2070 batches through
  runtime-v5
- Immediate post-EOF flush pacing with flush compute retained in finalization latency

## Remaining Gate blocker

- Make terminal output robust enough to reach 0.95 raw exact agreement on unseen clips without
  losing runtime-v5's passing finalization latency. Runtime-v5 reached only 0.65 exact agreement.
- Any further implementation candidate requires a new protocol and new unseen engineering input;
  the v3 protocol permits no additional official run.

The synthetic endpoint test had one permitted early trigger (1/20); natural conversational
endpoint data remains a Gate 14 limitation. Gate 4 must not be marked started while Gate 3 is
FAIL.
