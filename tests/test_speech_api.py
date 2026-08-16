import json
import wave
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from busan_lab.speech_api import SpeechAPISettings, create_speech_app
from busan_lab.speech_runtime import (
    ASRInference,
    SpeechQueueTimeout,
    TTSInference,
    UnavailableASRBackend,
    UnavailableTTSBackend,
    tensor_digest,
)
from tests.helpers import make_wav_bytes


class FakeASRBackend:
    model_version = "busan-asr-test"
    device = "cuda:0"

    def __init__(self, transcript: str = "뭐 하노") -> None:
        self.transcript = transcript
        self.loaded = True
        self.unavailable_reason = None
        self.observed_audio_contract: tuple[int, int, int] | None = None

    def load(self) -> None:
        self.loaded = True

    def transcribe(self, audio_path: Path) -> ASRInference:
        with wave.open(str(audio_path), "rb") as audio:
            self.observed_audio_contract = (
                audio.getframerate(),
                audio.getnchannels(),
                audio.getsampwidth(),
            )
        return ASRInference(transcript=self.transcript, latency_ms=2450.4)


class FakeTTSBackend:
    model_version = "busan-tts-test"
    loaded = True
    unavailable_reason = None

    def load(self) -> None:
        self.loaded = True

    def generate(self, *, sentence_id: str, text: str, voice: str) -> TTSInference:
        assert sentence_id == "busan-survival-001"
        assert text == "밥 묵었나?"
        assert voice == "busan-speaker-01"
        return TTSInference(
            audio_url="/v1/tts/audio/busan-survival-001.wav",
            cached=True,
            duration_ms=1840,
            sample_rate=24_000,
        )


class BusyASRBackend(FakeASRBackend):
    def transcribe(self, audio_path: Path) -> ASRInference:
        del audio_path
        raise SpeechQueueTimeout("fixture queue timeout")


class FakeTensor:
    dtype = "torch.float32"
    shape = (2,)

    def __init__(self) -> None:
        self.observed_view_dtype: object | None = None

    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def view(self, dtype: object):
        self.observed_view_dtype = dtype
        return self

    def numpy(self) -> np.ndarray:
        return np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint8)


def make_client(
    *,
    asr_backend: object | None = None,
    tts_backend: object | None = None,
    max_upload_bytes: int = 20 * 1024 * 1024,
) -> TestClient:
    settings = SpeechAPISettings(
        service_version="speech-api-test",
        git_commit="abc1234",
        max_upload_bytes=max_upload_bytes,
    )
    return TestClient(
        create_speech_app(
            asr_backend=asr_backend or FakeASRBackend(),  # type: ignore[arg-type]
            tts_backend=tts_backend or UnavailableTTSBackend("TTS fixture unavailable"),  # type: ignore[arg-type]
            settings=settings,
        )
    )


def post_attempt(
    client: TestClient,
    audio_bytes: bytes,
    *,
    target_text: str = "뭐 하노",
    filename: str = "attempt.wav",
    extra_data: dict[str, str] | None = None,
):
    data = {
        "sentence_id": "busan-survival-001",
        "target_text": target_text,
    }
    if extra_data:
        data.update(extra_data)
    return client.post(
        "/v1/practice/attempt",
        files={"audio": (filename, audio_bytes, "audio/wav")},
        data=data,
        headers={"X-Request-ID": "request-fixture-001"},
    )


def test_health_and_version_expose_runtime_state() -> None:
    client = make_client()

    health = client.get("/health")
    version = client.get("/version")

    assert health.status_code == 200
    assert health.json() == {
        "status": "degraded",
        "asr_loaded": True,
        "tts_loaded": False,
        "gpu_available": True,
    }
    assert version.json() == {
        "service_version": "speech-api-test",
        "asr_model_version": "busan-asr-test",
        "tts_model_version": "unavailable",
        "asr_mode": "raw",
        "git_commit": "abc1234",
    }
    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    assert {
        "/health",
        "/version",
        "/v1/practice/attempt",
        "/v1/tts",
    } <= set(openapi.json()["paths"])


def test_tensor_digest_uses_the_runtime_tensor_uint8_dtype() -> None:
    tensor = FakeTensor()
    uint8_dtype = object()

    digest = tensor_digest([("adapter.weight", tensor)], uint8_dtype=uint8_dtype)

    assert tensor.observed_view_dtype is uint8_dtype
    assert len(digest) == 64


def test_practice_attempt_converts_audio_and_returns_raw_contract() -> None:
    backend = FakeASRBackend()
    client = make_client(asr_backend=backend)

    response = post_attempt(client, make_wav_bytes(sample_rate=44_100, channels=2))

    assert response.status_code == 200, response.text
    assert backend.observed_audio_contract == (16_000, 1, 2)
    payload = response.json()
    assert payload["transcript"] == "뭐 하노"
    assert payload["raw_transcript"] == "뭐 하노"
    assert payload["sentence_match"] == 100
    assert payload["dialect_match"] == 100
    assert payload["focus_expression"] == "-노"
    assert payload["feedback"]["code"] == "MATCHED"
    assert payload["model_version"] == "busan-asr-test"
    assert payload["mode"] == "raw"
    assert payload["postprocessing_applied"] is False
    assert payload["latency_ms"] == 2450
    assert payload["request_id"] == "request-fixture-001"
    assert response.headers["X-Request-ID"] == "request-fixture-001"


def test_practice_attempt_keeps_raw_mismatch_and_scores_dialect() -> None:
    client = make_client(asr_backend=FakeASRBackend("뭐 하냐"))

    response = post_attempt(
        client,
        make_wav_bytes(),
        extra_data={
            "focus_expression": "-노",
            "normalized_focus_forms": json.dumps(["-냐"]),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["raw_transcript"] == "뭐 하냐"
    assert payload["transcript"] == "뭐 하냐"
    assert payload["dialect_match"] == 0
    assert payload["feedback"]["code"] == "DIALECT_EXPRESSION_MISMATCH"


def test_empty_invalid_short_long_and_silent_audio_are_safe_errors() -> None:
    client = make_client()
    cases = (
        (b"", "empty.wav", "UNSUPPORTED_AUDIO"),
        (b"not audio", "invalid.bin", "UNSUPPORTED_AUDIO"),
        (make_wav_bytes(duration_seconds=0.2), "short.wav", "AUDIO_TOO_SHORT"),
        (make_wav_bytes(duration_seconds=15.1), "long.wav", "AUDIO_TOO_LONG"),
        (make_wav_bytes(amplitude=0), "silent.wav", "NO_SPEECH"),
    )

    for audio_bytes, filename, expected_code in cases:
        response = post_attempt(client, audio_bytes, filename=filename)
        assert response.status_code in {415, 422}, response.text
        assert response.json()["error"]["code"] == expected_code
        assert response.json()["error"]["request_id"] == "request-fixture-001"
        assert "Traceback" not in response.text


def test_upload_size_limit_is_enforced_before_inference() -> None:
    client = make_client(max_upload_bytes=100)

    response = post_attempt(client, make_wav_bytes())

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "UPLOAD_FAILED"
    assert response.json()["error"]["retryable"] is False


def test_unavailable_asr_returns_versioned_service_error() -> None:
    client = make_client(
        asr_backend=UnavailableASRBackend("checkpoint missing", model_version="busan-asr-candidate")
    )

    response = post_attempt(client, make_wav_bytes())

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "ASR_UNAVAILABLE"
    assert response.json()["error"]["retryable"] is True


def test_busy_single_gpu_queue_returns_rate_limited() -> None:
    client = make_client(asr_backend=BusyASRBackend())

    response = post_attempt(client, make_wav_bytes())

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "RATE_LIMITED"
    assert response.json()["error"]["retryable"] is True


def test_tts_unavailable_and_success_contracts() -> None:
    unavailable = make_client()

    unavailable_response = unavailable.post(
        "/v1/tts",
        json={
            "sentence_id": "busan-survival-001",
            "text": "밥 묵었나?",
            "voice": "busan-speaker-01",
        },
    )

    assert unavailable_response.status_code == 503
    assert unavailable_response.json()["error"]["code"] == "TTS_UNAVAILABLE"

    available = make_client(tts_backend=FakeTTSBackend())
    response = available.post(
        "/v1/tts",
        json={
            "sentence_id": "busan-survival-001",
            "text": "밥 묵었나?",
            "voice": "busan-speaker-01",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "audio_url": "/v1/tts/audio/busan-survival-001.wav",
        "cached": True,
        "duration_ms": 1840,
        "sample_rate": 24_000,
        "model_version": "busan-tts-test",
    }
