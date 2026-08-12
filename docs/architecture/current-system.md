# Current system architecture — 2026-08-12

```text
Browser file upload / local dataset
  → FastAPI Audio Lab
  → source-byte storage + FFprobe validation
  → FFmpeg master and ASR/pronunciation/TTS derivatives
  → Utterance + consent + lineage schemas
  ├─ analysis: waveform / log-Mel / exploratory F0
  ├─ evaluation: provider-neutral Surface ASR + CER/error export
  ├─ research: benchmark / experiment / human review
  └─ training data: import / review / leakage validation / export

Frozen Gate 2 Nemotron adapter (.nemo)
  → WSL CUDA external runner
  → 320ms cache-aware FastConformer-RNNT steps
  → immediate post-EOF 320ms zero-PCM end flush (offline comparator remains unpadded)
  → partial transcript
  → stable prefix + unstable suffix tracker
  → final agreement / latency / stability trace
  → local raw evidence + tracked redacted aggregate
```

## Boundaries

- `src/busan_lab/` is model-neutral and runs without CUDA or NeMo.
- `external_inference/nemotron_3_5/` owns the NVIDIA/NeMo dependency and GPU commands.
- `data/lab/` holds licensed/private audio, labels, model weights, and raw runtime outputs and is
  excluded from Git.
- `artifacts/` holds non-sensitive criteria, aggregate evidence, hashes, and decisions.

## Streaming lifecycle

```text
active session
  → observe monotonic audio chunk
  → update cache in NeMo runtime
  → publish partial event
  → stabilize common prefix over three observations
  ├─ explicit end-of-input → zero-PCM cache flush → final event → finalized
  ├─ cancel → cancelled
  └─ reset → new generation, empty history
```

The GPU batch runner exercises active-to-finalized, cancellation, reset/reuse, state release, and
sequential-session memory behavior. Its endpoint probe is a separate deterministic RMS baseline
over source audio plus appended silence.

Runtime-v5 passes finalization-lag p95 at 246.17ms on its disjoint engineering set but remains a
Gate 3 failure because raw exact agreement was 0.65 against the frozen 0.95 minimum.

## Deliberately absent

- Live microphone/WebRTC media gateway, VAD/AEC/noise suppression, and barge-in
- Production streaming API or browser partial-transcript UI
- Natural-conversation validated speech endpoint detector (only a synthetic silence probe exists)
- Calibrated streaming confidence
- Direct IPA, forced alignment, pronunciation diagnosis, dialect prosody, and TTS
- Multi-worker persistence locking, authentication, encryption, and production retention

These are future gates, not implied by the successful file-based streaming smoke.
