"""Experiment, stored prediction, human review, and A/B comparison contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import Field

from busan_lab.schemas.asr import ModelDescriptor
from busan_lab.schemas.common import SCHEMA_VERSION, StrictSchema, utc_now
from busan_lab.schemas.evaluation import EvaluationCaseResult


class ExperimentStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PredictionSource(StrEnum):
    MANUAL = "manual"
    PRECOMPUTED = "precomputed"
    MODEL_RUNNER = "model_runner"


class ReviewVerdict(StrEnum):
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"


class ExperimentRun(StrictSchema):
    """Immutable identity and model conditions for one Surface ASR experiment."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    experiment_id: str = Field(min_length=1, max_length=128)
    task: Literal["surface_asr"] = "surface_asr"
    model: ModelDescriptor
    benchmark_id: str | None = None
    benchmark_version: str | None = None
    hypothesis: str | None = None
    changed_variable: str | None = None
    status: ExperimentStatus = ExperimentStatus.COMPLETED
    created_at: datetime = Field(default_factory=utc_now)


class StoredPrediction(StrictSchema):
    """Persisted model result linked to an experiment and immutable audio."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    prediction_id: UUID = Field(default_factory=uuid4)
    experiment_id: str = Field(min_length=1, max_length=128)
    utterance_id: UUID
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: PredictionSource
    latency_ms: float | None = Field(default=None, ge=0)
    automatic_failure_candidates: tuple[str, ...] = ()
    evaluation: EvaluationCaseResult
    created_at: datetime = Field(default_factory=utc_now)


class HumanReview(StrictSchema):
    """Append-only human judgment of one stored prediction."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    review_id: UUID = Field(default_factory=uuid4)
    prediction_id: UUID
    utterance_id: UUID
    reviewer_id: str = Field(min_length=1, max_length=128)
    verdict: ReviewVerdict
    confirmed_failure_types: tuple[str, ...] = ()
    notes: str | None = Field(default=None, max_length=4000)
    created_at: datetime = Field(default_factory=utc_now)


class PredictionComparison(StrictSchema):
    """A/B comparison restricted to two predictions for the same utterance."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    utterance_id: UUID
    prediction_a: StoredPrediction
    prediction_b: StoredPrediction
    cer_delta_b_minus_a: float
    preservation_delta_b_minus_a: float
    overcorrection_delta_b_minus_a: float
    confidence_delta_b_minus_a: float | None
