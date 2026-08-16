"""Production-facing offline Speech API for the lesson application."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import Field
from starlette.concurrency import run_in_threadpool

from busan_lab.evaluation.metrics import (
    character_error_rate,
    evaluate_dialect_preservation,
)
from busan_lab.schemas.common import ReviewStatus, StrictSchema
from busan_lab.schemas.utterance import DialectExpressionLabel
from busan_lab.speech_runtime import (
    DEFAULT_ADAPTER_SHA256,
    DEFAULT_MODEL_PATH,
    DEFAULT_MODEL_SHA256,
    DEFAULT_MODEL_VERSION,
    ASRBackend,
    AudioContractError,
    NemoOfflineASRBackend,
    PracticeAudioPreprocessor,
    SpeechQueueTimeout,
    SpeechRuntimeError,
    TTSBackend,
    UnavailableASRBackend,
    UnavailableTTSBackend,
    git_commit_from_environment,
)

DEFAULT_SERVICE_VERSION = "speech-api-dev-20260816"
KNOWN_DIALECT_EXPRESSIONS = (
    "-아이가",
    "-심더",
    "-카이",
    "-데이",
    "-노",
    "-나",
    "-가",
    "-예",
    "-제",
    "-마",
)


@dataclass(frozen=True, slots=True)
class SpeechAPISettings:
    service_version: str = DEFAULT_SERVICE_VERSION
    git_commit: str = "unknown"
    max_upload_bytes: int = 20 * 1024 * 1024
    min_duration_ms: float = 300
    max_duration_ms: float = 15_000
    eager_load: bool = False

    @classmethod
    def from_environment(cls) -> SpeechAPISettings:
        return cls(
            service_version=os.getenv("BUSAN_SPEECH_SERVICE_VERSION", DEFAULT_SERVICE_VERSION),
            git_commit=git_commit_from_environment(),
            max_upload_bytes=int(os.getenv("BUSAN_SPEECH_MAX_UPLOAD_BYTES", 20 * 1024 * 1024)),
            min_duration_ms=float(os.getenv("BUSAN_SPEECH_MIN_DURATION_MS", 300)),
            max_duration_ms=float(os.getenv("BUSAN_SPEECH_MAX_DURATION_MS", 15_000)),
            eager_load=_environment_bool("BUSAN_SPEECH_EAGER_LOAD", default=False),
        )


class TTSRequest(StrictSchema):
    sentence_id: str = Field(min_length=1, max_length=128)
    text: str | None = Field(default=None, max_length=300)
    voice: str = Field(default="busan-speaker-01", min_length=1, max_length=128)


class SpeechAPIError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int,
        retryable: bool,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable


def create_speech_app(
    *,
    asr_backend: ASRBackend | None = None,
    tts_backend: TTSBackend | None = None,
    settings: SpeechAPISettings | None = None,
    preprocessor: PracticeAudioPreprocessor | None = None,
) -> FastAPI:
    resolved_settings = settings or SpeechAPISettings.from_environment()
    resolved_asr = asr_backend or _default_asr_backend()
    resolved_tts = tts_backend or UnavailableTTSBackend(
        "No licensed TTS checkpoint and consented speaker dataset are configured."
    )
    resolved_preprocessor = preprocessor or PracticeAudioPreprocessor(
        min_duration_ms=resolved_settings.min_duration_ms,
        max_duration_ms=resolved_settings.max_duration_ms,
    )
    startup_errors: dict[str, str] = {}

    @asynccontextmanager
    async def lifespan(_application: FastAPI) -> AsyncIterator[None]:
        if resolved_settings.eager_load:
            for name, backend in (("asr", resolved_asr), ("tts", resolved_tts)):
                if backend.loaded:
                    continue
                try:
                    await run_in_threadpool(backend.load)
                except SpeechRuntimeError as error:
                    startup_errors[name] = str(error)
        yield

    application = FastAPI(
        title="Busan Lesson Speech API",
        version=resolved_settings.service_version,
        description=(
            "Offline raw Surface ASR for lesson practice. Gate metrics never include "
            "lesson post-processing."
        ),
        lifespan=lifespan,
    )
    application.state.asr_backend = resolved_asr
    application.state.tts_backend = resolved_tts
    application.state.speech_settings = resolved_settings
    application.state.startup_errors = startup_errors

    @application.middleware("http")
    async def attach_request_id(request: Request, call_next: Any) -> Any:
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        request.state.request_id = request_id
        try:
            response = await call_next(request)
        except Exception:
            response = _error_response(
                request_id=request_id,
                code="INTERNAL_ERROR",
                message="The speech service encountered an internal error.",
                retryable=True,
                status_code=500,
            )
        response.headers["X-Request-ID"] = request_id
        return response

    @application.exception_handler(SpeechAPIError)
    async def handle_speech_error(request: Request, error: SpeechAPIError) -> JSONResponse:
        return _error_response(
            request_id=_request_id(request),
            code=error.code,
            message=error.message,
            retryable=error.retryable,
            status_code=error.status_code,
        )

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, _error: RequestValidationError
    ) -> JSONResponse:
        return _error_response(
            request_id=_request_id(request),
            code="UPLOAD_FAILED",
            message="The request does not match the Speech API contract.",
            retryable=False,
            status_code=422,
        )

    @application.get("/health")
    async def health() -> dict[str, object]:
        asr_loaded = resolved_asr.loaded
        tts_loaded = resolved_tts.loaded
        return {
            "status": "ok" if asr_loaded and tts_loaded else "degraded",
            "asr_loaded": asr_loaded,
            "tts_loaded": tts_loaded,
            "gpu_available": bool(
                getattr(resolved_asr, "device", "").startswith("cuda") and asr_loaded
            ),
        }

    @application.get("/version")
    async def version() -> dict[str, object]:
        return {
            "service_version": resolved_settings.service_version,
            "asr_model_version": resolved_asr.model_version,
            "tts_model_version": resolved_tts.model_version,
            "asr_mode": "raw",
            "git_commit": resolved_settings.git_commit,
        }

    @application.post("/v1/practice/attempt")
    async def practice_attempt(
        request: Request,
        audio: Annotated[UploadFile, File(description="Practice recording")],
        sentence_id: Annotated[str, Form(min_length=1, max_length=128)],
        target_text: Annotated[str, Form(min_length=1, max_length=300)],
        focus_expression: Annotated[str | None, Form(max_length=64)] = None,
        normalized_focus_forms: Annotated[str, Form(max_length=500)] = "[]",
    ) -> dict[str, object]:
        if not resolved_asr.loaded:
            raise SpeechAPIError(
                "ASR_UNAVAILABLE",
                "The ASR model is not available.",
                status_code=503,
                retryable=True,
            )
        normalized_forms = _parse_normalized_forms(normalized_focus_forms)
        resolved_focus = focus_expression or _infer_focus_expression(target_text)
        with tempfile.TemporaryDirectory(prefix="busan-speech-") as temporary_directory:
            temporary_root = Path(temporary_directory)
            suffix = Path(audio.filename or "upload.audio").suffix[:10] or ".audio"
            source = temporary_root / f"upload{suffix}"
            prepared = temporary_root / "asr-16k-mono.wav"
            await _save_upload(
                audio,
                source,
                max_upload_bytes=resolved_settings.max_upload_bytes,
            )
            try:
                await run_in_threadpool(resolved_preprocessor.prepare, source, prepared)
            except AudioContractError as error:
                raise _audio_error(error) from error
            try:
                inference = await run_in_threadpool(resolved_asr.transcribe, prepared)
            except SpeechQueueTimeout as error:
                raise SpeechAPIError(
                    "RATE_LIMITED",
                    "The single-GPU inference queue is busy. Please retry.",
                    status_code=429,
                    retryable=True,
                ) from error
            except SpeechRuntimeError as error:
                raise SpeechAPIError(
                    "ASR_INFERENCE_FAILED",
                    "ASR inference failed safely. Please retry.",
                    status_code=503,
                    retryable=True,
                ) from error

        cer, _edits = character_error_rate(target_text, inference.transcript)
        sentence_match = round(max(0.0, 1.0 - min(cer, 1.0)) * 100)
        dialect_match = 100
        if resolved_focus:
            label = DialectExpressionLabel(
                surface_form=_matching_form(resolved_focus),
                normalized_forms=tuple(_matching_form(value) for value in normalized_forms),
                status=ReviewStatus.APPROVED,
            )
            dialect_metric = evaluate_dialect_preservation(inference.transcript, (label,))
            dialect_match = round(dialect_metric.preservation_rate * 100)
        feedback = _feedback(
            sentence_match=sentence_match,
            dialect_match=dialect_match,
            focus_expression=resolved_focus,
        )
        return {
            "attempt_id": str(uuid4()),
            "request_id": _request_id(request),
            "sentence_id": sentence_id,
            "transcript": inference.transcript,
            "raw_transcript": inference.transcript,
            "target_text": target_text,
            "sentence_match": sentence_match,
            "dialect_match": dialect_match,
            "focus_expression": resolved_focus,
            "feedback": feedback,
            "model_version": resolved_asr.model_version,
            "mode": "raw",
            "postprocessing_applied": False,
            "latency_ms": round(inference.latency_ms),
        }

    @application.post("/v1/tts")
    async def tts(payload: TTSRequest) -> dict[str, object]:
        if not resolved_tts.loaded:
            raise SpeechAPIError(
                "TTS_UNAVAILABLE",
                "The TTS release candidate is not configured.",
                status_code=503,
                retryable=True,
            )
        if not payload.text:
            raise SpeechAPIError(
                "INVALID_SENTENCE_ID",
                "No text is registered for the requested sentence ID.",
                status_code=404,
                retryable=False,
            )
        try:
            generated = await run_in_threadpool(
                resolved_tts.generate,
                sentence_id=payload.sentence_id,
                text=payload.text,
                voice=payload.voice,
            )
        except SpeechRuntimeError as error:
            raise SpeechAPIError(
                "TTS_GENERATION_FAILED",
                "TTS generation failed safely. Please retry.",
                status_code=503,
                retryable=True,
            ) from error
        return {
            "audio_url": generated.audio_url,
            "cached": generated.cached,
            "duration_ms": generated.duration_ms,
            "sample_rate": generated.sample_rate,
            "model_version": resolved_tts.model_version,
        }

    return application


def _default_asr_backend() -> ASRBackend:
    model_path = Path(os.getenv("BUSAN_ASR_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    if not model_path.is_file():
        return UnavailableASRBackend(
            f"Checkpoint is missing: {model_path}",
            model_version=os.getenv("BUSAN_ASR_MODEL_VERSION", DEFAULT_MODEL_VERSION),
        )
    return NemoOfflineASRBackend(
        model_path=model_path,
        expected_model_sha256=os.getenv("BUSAN_ASR_MODEL_SHA256", DEFAULT_MODEL_SHA256),
        expected_adapter_sha256=os.getenv("BUSAN_ASR_ADAPTER_SHA256", DEFAULT_ADAPTER_SHA256),
        model_version=os.getenv("BUSAN_ASR_MODEL_VERSION", DEFAULT_MODEL_VERSION),
        device=os.getenv("BUSAN_ASR_DEVICE", "cuda:0"),
    )


async def _save_upload(upload: UploadFile, destination: Path, *, max_upload_bytes: int) -> None:
    total_bytes = 0
    try:
        with destination.open("wb") as stream:
            while chunk := await upload.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > max_upload_bytes:
                    raise SpeechAPIError(
                        "UPLOAD_FAILED",
                        "The audio upload is too large.",
                        status_code=413,
                        retryable=False,
                    )
                stream.write(chunk)
    except SpeechAPIError:
        raise
    except OSError as error:
        raise SpeechAPIError(
            "UPLOAD_FAILED",
            "The audio upload could not be stored temporarily.",
            status_code=500,
            retryable=True,
        ) from error
    finally:
        await upload.close()
    if total_bytes == 0:
        raise SpeechAPIError(
            "UNSUPPORTED_AUDIO",
            "The uploaded file is empty.",
            status_code=415,
            retryable=False,
        )


def _audio_error(error: AudioContractError) -> SpeechAPIError:
    status_by_code = {
        "UNSUPPORTED_AUDIO": 415,
        "AUDIO_TOO_SHORT": 422,
        "AUDIO_TOO_LONG": 422,
        "NO_SPEECH": 422,
        "UPLOAD_FAILED": 500,
    }
    return SpeechAPIError(
        error.code,
        str(error),
        status_code=status_by_code.get(error.code, 422),
        retryable=error.code in {"AUDIO_TOO_SHORT", "NO_SPEECH", "UPLOAD_FAILED"},
    )


def _parse_normalized_forms(raw: str) -> tuple[str, ...]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SpeechAPIError(
            "UPLOAD_FAILED",
            "normalized_focus_forms must be a JSON string array.",
            status_code=422,
            retryable=False,
        ) from error
    if not isinstance(parsed, list) or any(not isinstance(value, str) for value in parsed):
        raise SpeechAPIError(
            "UPLOAD_FAILED",
            "normalized_focus_forms must be a JSON string array.",
            status_code=422,
            retryable=False,
        )
    return tuple(value for value in parsed if value.strip())


def _infer_focus_expression(target_text: str) -> str | None:
    for expression in KNOWN_DIALECT_EXPRESSIONS:
        if _matching_form(expression) in target_text:
            return expression
    return None


def _matching_form(expression: str) -> str:
    return expression[1:] if expression.startswith("-") else expression


def _feedback(
    *, sentence_match: int, dialect_match: int, focus_expression: str | None
) -> dict[str, str]:
    if sentence_match == 100 and dialect_match == 100:
        message = "Great. Your sentence matched the target."
        if focus_expression:
            message = f"Great. You kept the Busan expression '{focus_expression}'."
        return {"code": "MATCHED", "message_en": message}
    if focus_expression and dialect_match < 100:
        return {
            "code": "DIALECT_EXPRESSION_MISMATCH",
            "message_en": f"Try again and keep the Busan expression '{focus_expression}'.",
        }
    return {
        "code": "SENTENCE_MISMATCH",
        "message_en": "Listen to the reference and try the sentence again.",
    }


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", uuid4()))


def _error_response(
    *,
    request_id: str,
    code: str,
    message: str,
    retryable: bool,
    status_code: int,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "retryable": retryable,
                "request_id": request_id,
            }
        },
    )


def _environment_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


app = create_speech_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("busan_lab.speech_api:app", host="0.0.0.0", port=8000, reload=False)
