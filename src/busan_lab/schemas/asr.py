"""Surface ASR adapter contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from busan_lab.schemas.common import SCHEMA_VERSION, StrictSchema


class ModelDescriptor(StrictSchema):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    model_provider: str | None = Field(default=None, min_length=1)
    model_family: str | None = Field(default=None, min_length=1)
    decoder_type: str | None = Field(default=None, min_length=1)
    target_language: str | None = Field(default=None, min_length=1)
    fine_tuned: bool | None = None
    checkpoint_identifier: str | None = Field(default=None, min_length=1)
    # Compatibility field for existing experiment evidence.
    checkpoint: str | None = None
    tokenizer_version: str | None = None
    config_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class ASRSegment(StrictSchema):
    text: str
    start_ms: float = Field(ge=0)
    end_ms: float = Field(gt=0)
    confidence: float = Field(ge=0, le=1)


class SurfaceASRResult(StrictSchema):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    surface_text: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_supported: bool = True
    latency_ms: float = Field(ge=0)
    model: ModelDescriptor
    segments: tuple[ASRSegment, ...] = ()

    @model_validator(mode="after")
    def confidence_matches_support_flag(self) -> SurfaceASRResult:
        if self.confidence_supported != (self.confidence is not None):
            raise ValueError("confidence_supported must match whether confidence is present")
        return self


class PrecomputedPrediction(StrictSchema):
    experiment_id: str = Field(min_length=1, max_length=128)
    benchmark_id: str | None = Field(default=None, min_length=1)
    benchmark_version: str | None = Field(default=None, min_length=1)
    utterance_id: str
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    device: str | None = Field(default=None, min_length=1)
    inference_timestamp: datetime | None = None
    result: SurfaceASRResult
