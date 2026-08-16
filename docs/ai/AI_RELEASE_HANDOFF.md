# AI Release Handoff

This contract is ready for frontend/backend integration against a mock or the ASR service.
The ASR path is implemented and GPU-smoke-tested. TTS remains unavailable until a licensed
model and consented Busan speaker data are provided; clients must handle `TTS_UNAVAILABLE`.

## Service

- Local API Base URL: `http://localhost:8000`
- Service version: `speech-api-dev-20260816`
- ASR model version: `busan-asr-gate2-pass-20260812`
- ASR mode: `raw`
- OpenAPI JSON: `GET /openapi.json`
- Swagger UI: `GET /docs`
- Recommended client timeout: 20 seconds after service readiness
- Startup allowance: 300 seconds for cold model load
- Retry policy: retry `429` and retryable `503` once with jitter; do not retry validation
  errors without changing the request

## Audio Contract

- Request: `multipart/form-data`
- Maximum upload: 20 MiB
- Duration: 300 ms through 15 seconds
- Accepted when FFmpeg can decode the audio stream. Integration targets are WAV, MP3,
  M4A/AAC, OGG, FLAC, WebM, and CAF.
- Server normalization: 16 kHz, mono, PCM16 WAV in request-scoped temporary storage
- The source upload is never overwritten or persisted by the practice endpoint.
- Empty, undecodable, too short, too long, and silent audio are rejected before inference.

## `GET /health`

Response:

```json
{
  "status": "degraded",
  "asr_loaded": true,
  "tts_loaded": false,
  "gpu_available": true
}
```

`degraded` is expected until the TTS runtime exists. Deployment readiness for ASR should
check `asr_loaded`, not only the top-level status.

## `GET /version`

Response:

```json
{
  "service_version": "speech-api-dev-20260816",
  "asr_model_version": "busan-asr-gate2-pass-20260812",
  "tts_model_version": "unavailable",
  "asr_mode": "raw",
  "git_commit": "da5265a"
}
```

## `POST /v1/practice/attempt`

Request fields:

| Field | Type | Required | Notes |
|---|---|---:|---|
| `audio` | file | yes | Practice recording |
| `sentence_id` | string | yes | Maximum 128 characters |
| `target_text` | string | yes | Busan surface form, maximum 300 characters |
| `focus_expression` | string | no | Example: `-노`; inferred from known expressions when absent |
| `normalized_focus_forms` | JSON string array | no | Example: `["-냐"]`; used only to classify overcorrection |

Example:

```bash
curl -X POST http://localhost:8000/v1/practice/attempt \
  -H 'X-Request-ID: mobile-attempt-001' \
  -F 'audio=@attempt.m4a' \
  -F 'sentence_id=busan-survival-001' \
  -F 'target_text=뭐 하노' \
  -F 'focus_expression=-노' \
  -F 'normalized_focus_forms=["-냐"]'
```

Success response:

```json
{
  "attempt_id": "2bb9fd62-e1ea-4cb2-ae77-1b55ec759647",
  "request_id": "mobile-attempt-001",
  "sentence_id": "busan-survival-001",
  "transcript": "뭐 하노",
  "raw_transcript": "뭐 하노",
  "target_text": "뭐 하노",
  "sentence_match": 100,
  "dialect_match": 100,
  "focus_expression": "-노",
  "feedback": {
    "code": "MATCHED",
    "message_en": "Great. You kept the Busan expression '-노'."
  },
  "model_version": "busan-asr-gate2-pass-20260812",
  "mode": "raw",
  "postprocessing_applied": false,
  "latency_ms": 4338
}
```

The current service never changes `raw_transcript`. If a future lesson-constrained mode is
added, `raw_transcript`, `transcript`, `mode`, and `postprocessing_applied` must remain
separate.

## `POST /v1/tts`

Request:

```json
{
  "sentence_id": "busan-survival-001",
  "text": "밥 묵었나?",
  "voice": "busan-speaker-01"
}
```

Planned success response contract:

```json
{
  "audio_url": "/v1/tts/audio/busan-survival-001.wav",
  "cached": true,
  "duration_ms": 1840,
  "sample_rate": 24000,
  "model_version": "busan-tts-rc1-20260824"
}
```

Current real response is HTTP 503 `TTS_UNAVAILABLE`. Do not treat the planned response as
evidence that a TTS model exists.

## Error Contract

Every controlled error uses this envelope and returns the request ID in both the body and
`X-Request-ID` response header:

```json
{
  "error": {
    "code": "AUDIO_TOO_SHORT",
    "message": "The recording is too short to analyze.",
    "retryable": true,
    "request_id": "mobile-attempt-001"
  }
}
```

| Code | HTTP | Retry | Meaning |
|---|---:|---:|---|
| `UNSUPPORTED_AUDIO` | 415 | no | Empty or undecodable audio |
| `AUDIO_TOO_SHORT` | 422 | yes | Less than 300 ms |
| `AUDIO_TOO_LONG` | 422 | no | More than 15 seconds |
| `NO_SPEECH` | 422 | yes | Empty/silent decoded signal |
| `UPLOAD_FAILED` | 413/422/500 | depends | Oversize, malformed, or temporary storage/converter failure |
| `ASR_UNAVAILABLE` | 503 | yes | ASR was not loaded |
| `ASR_INFERENCE_FAILED` | 503 | yes | Controlled model inference failure |
| `TTS_UNAVAILABLE` | 503 | yes | No TTS runtime is configured |
| `TTS_GENERATION_FAILED` | 503 | yes | Controlled TTS generation failure |
| `INVALID_SENTENCE_ID` | 404 | no | TTS text lookup unavailable for the ID |
| `RATE_LIMITED` | 429 | yes | Single-GPU queue deadline exceeded |
| `INTERNAL_ERROR` | 500 | yes | Sanitized unexpected server failure |

No stack trace is returned to the client.

## Mock Fixtures

- Contract tests: `tests/test_speech_api.py`
- Valid generated WAV helper: `tests/helpers.py::make_wav_bytes`
- Real non-test ASR smoke result:
  `artifacts/release/speech-api/asr-single-file-smoke-20260816.json`
- Real HTTP GPU smoke result:
  `artifacts/release/speech-api/http-end-to-end-smoke-20260816.json`
- 50-file Train stability result:
  `artifacts/release/speech-api/asr-50-sequential-stability-20260816.json`
- Tracked non-sensitive aggregate:
  `artifacts/release/speech-api/verification-summary-20260816.json`
- Expected API examples: this document and FastAPI `/openapi.json`

The three detailed results are local-only because they contain licensed Train identities,
paths, hashes, or transcripts. Do not attach them to a public issue or commit them. Use the
tracked aggregate for normal handoff.

## Mobile Adapter Mapping Required

The current uncommitted mobile types are a temporary mock contract, not the wire format.
Do not send a JSON `PracticeAttemptInput` directly to the server. The real mobile service
must upload multipart audio and map the response:

| Mobile field | Speech API wire field | Conversion |
|---|---|---|
| `audioUri` | `audio` | open local URI and append as multipart file |
| `sentenceId` | `sentence_id` | rename |
| `targetSurfaceText` | `target_text` | rename; preserve Busan surface form |
| `focusExpressions[0]` | `focus_expression` | release scope uses one primary expression |
| `attemptId` | `attempt_id` | rename |
| `targetSurfaceText` | `target_text` | rename |
| `sentenceMatch` | `sentence_match` | divide integer 0-100 by 100 for current UI type |
| `dialectMatch` | `dialect_match` | divide integer 0-100 by 100 for current UI type |
| `modelVersion` | `model_version` | rename |
| `processingTimeMs` | `latency_ms` | rename |
| `feedbackMessage` | `feedback.message_en` | select English message |
| `outcome` | derived | map feedback code and thresholds in the adapter |
| `createdAt` | not returned | add client receipt timestamp or extend the server contract |

`preservedExpressions` and `missedExpressions` are not yet returned by the wire contract.
Either extend the server response before freeze or derive them deterministically from the
focus expression in the mobile adapter. TTS mobile input currently has only `text` and
`speed`; the wire request additionally requires `sentence_id` and `voice`. That mapping
also remains integration work.

## Local Start

Contract-only environment, without NeMo/TTS:

```powershell
wsl.exe bash -lc 'cd /mnt/c/Users/ab409/orca/projects/wreckfish && PYTHONPATH=src .venv/bin/python -m uvicorn busan_lab.speech_api:app --host 0.0.0.0 --port 8000'
```

Real ASR requires the NeMo environment and `BUSAN_SPEECH_EAGER_LOAD=true`. Copy
`.env.speech.example` to `.env.speech`, set the current commit, and export the values before
starting one Uvicorn worker.

## Docker

```powershell
Copy-Item -LiteralPath .env.speech.example -Destination .env.speech
docker compose --env-file .env.speech -f docker-compose.speech.yml build
docker compose --env-file .env.speech -f docker-compose.speech.yml up
```

The checkpoint is mounted read-only and is never copied into the image or committed to Git.
Docker build/run is currently UNVERIFIED because the local Docker daemon was stopped.

## Known Limitations

- TTS is unavailable.
- This is offline, non-streaming ASR by release scope.
- The current mode is raw and does not use lesson context biasing or post-processing.
- Only one GPU inference runs at a time.
- Cold loading took about 193 seconds in the current WSL environment.
- iPhone codec fixtures and a Docker GPU run remain before freeze.
- General-purpose Busan speech outside lesson-length recordings is not claimed.
