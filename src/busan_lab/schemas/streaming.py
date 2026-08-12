"""Gate 3 Streaming ASR contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from busan_lab.schemas.asr import ModelDescriptor
from busan_lab.schemas.common import SCHEMA_VERSION, StrictSchema


class StreamingSessionStatus(StrEnum):
    ACTIVE = "active"
    FINALIZED = "finalized"
    CANCELLED = "cancelled"


class Gate3Decision(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"


StreamingText = Annotated[str, StringConstraints(strip_whitespace=False)]


class StreamingTranscriptEvent(StrictSchema):
    """One publishable partial or final transcript from a streaming session."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    session_id: str = Field(min_length=1, max_length=128)
    generation: int = Field(ge=0)
    sequence: int = Field(ge=0)
    partial: bool
    stable_prefix: StreamingText
    unstable_suffix: StreamingText
    transcript: StreamingText
    stability: float = Field(ge=0, le=1)
    audio_start_ms: float = Field(ge=0)
    audio_end_ms: float = Field(ge=0)
    emitted_at_ms: float = Field(ge=0)
    inference_latency_ms: float = Field(ge=0)
    endpoint_detected: bool = False
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_supported: bool = False
    model: ModelDescriptor

    @model_validator(mode="after")
    def validate_event_boundaries(self) -> StreamingTranscriptEvent:
        if self.audio_end_ms < self.audio_start_ms:
            raise ValueError("audio_end_ms must be greater than or equal to audio_start_ms")
        if self.transcript != self.stable_prefix + self.unstable_suffix:
            raise ValueError("stable_prefix and unstable_suffix must exactly form transcript")
        if self.confidence_supported != (self.confidence is not None):
            raise ValueError("confidence_supported must match whether confidence is present")
        if not self.partial and self.unstable_suffix:
            raise ValueError("a final event cannot retain an unstable suffix")
        return self


class StreamingTraceMetrics(StrictSchema):
    """Model-neutral metrics computed from one completed streaming trace."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    session_id: str = Field(min_length=1, max_length=128)
    generation: int = Field(ge=0)
    event_count: int = Field(ge=0)
    partial_event_count: int = Field(ge=0)
    endpoint_count: int = Field(ge=0)
    audio_duration_ms: float = Field(ge=0)
    first_partial_latency_ms: float | None = Field(default=None, ge=0)
    final_latency_ms: float | None = Field(default=None, ge=0)
    mean_chunk_inference_latency_ms: float | None = Field(default=None, ge=0)
    stable_prefix_observations: int = Field(ge=0)
    stable_prefix_violations: int = Field(ge=0)
    partial_stability: float = Field(ge=0, le=1)
    final_transcript: str
    surface_final_cer: float | None = Field(default=None, ge=0)
    final_agreement_with_surface_asr: float | None = Field(default=None, ge=0, le=1)
    peak_device_memory_bytes: int | None = Field(default=None, ge=0)


class Gate3Criteria(StrictSchema):
    """Frozen machine-readable Gate 3 engineering thresholds."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    criteria_id: str = Field(min_length=1)
    status: Literal["frozen_before_batch_outputs"]
    frozen_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_case_count: int = Field(gt=0)
    minimum_speaker_count: int = Field(gt=0)
    maximum_empty_final_count: int = Field(ge=0)
    minimum_nonempty_partial_rate: float = Field(ge=0, le=1)
    minimum_mean_partial_stability: float = Field(ge=0, le=1)
    maximum_trace_retraction_rate: float = Field(ge=0, le=1)
    minimum_exact_surface_agreement_rate: float = Field(ge=0, le=1)
    maximum_surface_cer: float = Field(ge=0)
    maximum_warm_first_partial_p95_ms: float = Field(gt=0)
    maximum_finalization_lag_p95_ms: float = Field(gt=0)
    maximum_post_warmup_chunk_inference_p95_ms: float = Field(gt=0)
    maximum_trace_realtime_factor_p95: float = Field(gt=0)
    minimum_synthetic_endpoint_f1: float = Field(ge=0, le=1)
    maximum_endpoint_early_trigger_rate: float = Field(ge=0, le=1)
    maximum_endpoint_delay_p95_ms: float = Field(gt=0)
    minimum_cancellation_correctness: float = Field(ge=0, le=1)
    minimum_reset_correctness: float = Field(ge=0, le=1)
    maximum_allocated_memory_growth_bytes: int = Field(ge=0)
    maximum_reserved_memory_growth_bytes: int = Field(ge=0)
    expected_adapter_module_count: int = Field(gt=0)
    require_every_adapter_called: bool
    require_every_session_state_released: bool
    confidence_policy: str = Field(min_length=1)
    quality_generalization_claim_allowed: bool
    endpoint_evidence_scope: str = Field(min_length=1)


class Gate3Evidence(StrictSchema):
    """Aggregate evidence from one frozen Gate 3 batch run."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    criteria_id: str = Field(min_length=1)
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_count: int = Field(ge=0)
    unique_case_count: int = Field(ge=0)
    speaker_count: int = Field(ge=0)
    complete_trace_count: int = Field(ge=0)
    empty_final_count: int = Field(ge=0)
    nonempty_partial_rate: float = Field(ge=0, le=1)
    mean_partial_stability: float = Field(ge=0, le=1)
    trace_retraction_rate: float = Field(ge=0, le=1)
    exact_surface_agreement_rate: float = Field(ge=0, le=1)
    aggregate_surface_cer: float = Field(ge=0)
    warm_first_partial_p95_ms: float = Field(ge=0)
    finalization_lag_p95_ms: float = Field(ge=0)
    post_warmup_chunk_inference_p95_ms: float = Field(ge=0)
    trace_realtime_factor_p95: float = Field(ge=0)
    synthetic_endpoint_precision: float = Field(ge=0, le=1)
    synthetic_endpoint_recall: float = Field(ge=0, le=1)
    synthetic_endpoint_f1: float = Field(ge=0, le=1)
    endpoint_early_trigger_rate: float = Field(ge=0, le=1)
    endpoint_delay_p95_ms: float = Field(ge=0)
    cancellation_correctness: float = Field(ge=0, le=1)
    reset_correctness: float = Field(ge=0, le=1)
    allocated_memory_growth_bytes: int = Field(ge=0)
    reserved_memory_growth_bytes: int = Field(ge=0)
    adapter_module_count: int = Field(ge=0)
    every_adapter_called: bool
    every_session_state_released: bool
    confidence_policy: str = Field(min_length=1)
    infrastructure_failures: tuple[str, ...] = ()


class Gate3Check(StrictSchema):
    name: str = Field(min_length=1)
    passed: bool
    observed: bool | int | float | str
    requirement: bool | int | float | str


class Gate3Assessment(StrictSchema):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    criteria_id: str = Field(min_length=1)
    decision: Gate3Decision
    checks: tuple[Gate3Check, ...]
    passed_checks: tuple[str, ...]
    failed_checks: tuple[str, ...]
