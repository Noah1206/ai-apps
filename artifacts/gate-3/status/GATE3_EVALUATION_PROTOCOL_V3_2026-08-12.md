# Gate 3 runtime-v5 evaluation protocol — frozen before inference

Status: **FROZEN 2026-08-12 before any v3 batch output**.

## Purpose

Runtime-v4's 320ms zero-PCM cache flush passed the raw exact-agreement requirement but its
real-time replay waited for the synthetic flush as if those samples still had to be captured.
That produced finalization-lag p95 `510.23ms`, exceeding the `500ms` budget by `10.23ms`.

Runtime-v5 retains exactly the same zero-PCM model input. Real source audio remains paced in real
time. Once explicit EOF is available, already-generated synthetic flush PCM is processed
immediately. Its GPU compute remains included in finalization lag and real-time factor, but no
synthetic capture wait is added. Offline comparison still receives only the original unpadded
audio, and the streaming path never reads or substitutes offline text.

## Frozen implementation

- Model SHA-256: `eb7d5112329504db72ffcb2c638e245e68f35801cda14a33682478d4a2ae85ee`
- Runtime candidate: `runtime-v5`
- Streaming runner SHA-256:
  `8443ba692bd32b0560b1fd25bd5b4c4d1395517c076fe2d8ba53571fc3052771`
- Batch runner SHA-256:
  `5e35ef4fad5ddb331544261415b663efa516bf09c91ac2823bcf9b5dbfb1295f`
- Benchmark builder SHA-256:
  `bf86c3cd68a65568db722c7d1090bfe66f49c468153124c157edadefd4c2d9a4`
- Runtime: Python 3.13.14, NeMo `3.1.0+6c57e73e83`, Torch `2.12.0+cu132`, CUDA GPU
- API: `CacheAwareStreamingAudioBuffer` + `conformer_stream_step`
- Decoder: RNNT `greedy_batch`; prompt `ko-KR`; language tags stripped
- Attention context: `[56, 3]`; compute float32 without AMP
- Source audio: 16kHz mono uncompressed PCM16
- Streaming-only end padding: exactly 320ms zero-valued PCM
- Pacing: source audio in real time; synthetic flush immediately after explicit EOF
- Offline comparison: original source audio only, with no padding
- Stable-prefix window: 3 observations

Source or configuration changes after the batch starts constitute another candidate and require
a new protocol version.

## Frozen input

- Benchmark ID: `gate3-streaming-engineering-v3`
- Local manifest SHA-256:
  `8bfbd54d0e1b631fc502773472a056b0b77891ae520cb6618c5d329eb2722b9b`
- Selection: 20 cases, 10 speakers, two cases per speaker, 2.5–7.0 seconds
- Seed: `gate3-engineering-v3-2026-08-12`
- Source manifest SHA-256:
  `6e5c23057e90b84c47141c976138b0faa6a9505342ed09cb70b4cd4fe45fe93c`
- Excluded v1 manifest SHA-256:
  `09b1685eaeb512ad1b9d5632ba40e6b829387bd5cf3ee0724038da2202d73b7a`
- Excluded v2 manifest SHA-256:
  `509bbdd97e73add605a51355ddaa572365acd1ab4134235bd24d5a1b338c37f3`
- Verified overlap with v1 and v2 combined: zero utterance IDs and zero audio hashes

The source is checkpoint Validation and therefore supports streaming engineering only. It does
not support a new ASR quality-generalization claim.

## Frozen decision

The machine-readable criteria are `gate3-criteria.v3.json`, SHA-256
`431cd7fae1635f9046ef97d0921a832e2f0d5982657713aec3fbaadb37f3f23f`. All 29 requirements
are unchanged from v1 and v2: 20/20 unique complete traces, 10 speakers, zero empty finals,
non-empty partial rate 1.0, mean stability at least 0.98, retraction rate at most 0.05, raw exact
offline agreement at least 0.95, aggregate CER at most 0.01, first-partial p95 at most 2500ms,
finalization-lag p95 at most 500ms, chunk-inference p95 at most 320ms, and real-time-factor p95
at most 1.15.

Synthetic endpoint F1 must be at least 0.90, early-trigger rate at most 0.10, and delay p95 at
most 1000ms. Cancellation and reset must both equal 1.0. Allocated and reserved memory growth
must remain within 64MiB and 256MiB. All 24 adapters must run for every session and all session
state must be released. Confidence remains explicitly unsupported; fabricated values are
prohibited.

Exactly one fixed v3 batch is permitted under this protocol. Its result is final whether PASS or
FAIL. Finalization lag is measured from the unpadded source boundary through immediate flush
inference and final publication. Real-time-factor denominators remain the unpadded source
duration.
