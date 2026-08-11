"""Evaluation and error-export contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import Field

from busan_lab.schemas.asr import ModelDescriptor
from busan_lab.schemas.common import SCHEMA_VERSION, ReviewStatus, StrictSchema, utc_now


class DialectMatchStatus(StrEnum):
    PRESERVED = "preserved"
    OVERCORRECTED = "overcorrected"
    MISSING = "missing"


class EditCounts(StrictSchema):
    substitutions: int = Field(ge=0)
    deletions: int = Field(ge=0)
    insertions: int = Field(ge=0)
    reference_characters: int = Field(ge=0)


class DialectExpressionResult(StrictSchema):
    surface_form: str
    normalized_forms: tuple[str, ...]
    label_status: ReviewStatus
    match_status: DialectMatchStatus
    matched_form: str | None = None


class DialectPreservationMetric(StrictSchema):
    preservation_rate: float = Field(ge=0, le=1)
    overcorrection_rate: float = Field(ge=0, le=1)
    reference_expression_count: int = Field(ge=0)
    approved_expression_count: int = Field(ge=0)
    results: tuple[DialectExpressionResult, ...]


class EvaluationCaseResult(StrictSchema):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    utterance_id: UUID
    reference_surface_text: str
    normalized_meaning: str | None
    hypothesis_surface_text: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    cer: float = Field(ge=0)
    edits: EditCounts
    dialect: DialectPreservationMetric
    high_confidence_wrong: bool
    model: ModelDescriptor


class AggregateMetrics(StrictSchema):
    utterance_count: int = Field(ge=0)
    cer: float = Field(ge=0)
    dialect_preservation_rate: float = Field(ge=0, le=1)
    context_overcorrection_rate: float = Field(ge=0, le=1)
    high_confidence_wrong_count: int = Field(ge=0)
    p50_latency_ms: float = Field(ge=0)
    p95_latency_ms: float = Field(ge=0)


class BaselineReport(StrictSchema):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    report_id: str
    experiment_id: str | None = None
    benchmark_id: str
    benchmark_version: str
    created_at: datetime = Field(default_factory=utc_now)
    model: ModelDescriptor
    metrics: AggregateMetrics
    cases: tuple[EvaluationCaseResult, ...]


class FailureTypeCount(StrictSchema):
    failure_type: str
    count: int = Field(ge=1)


class HumanReviewSummary(StrictSchema):
    reviewed_prediction_count: int = Field(ge=0)
    review_revision_count: int = Field(ge=0)
    confirmed_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    uncertain_count: int = Field(ge=0)
    confirmed_failure_type_counts: tuple[FailureTypeCount, ...] = ()
    confirmed_language_model_bias_count: int = Field(ge=0)


class ReviewedEvaluationCase(StrictSchema):
    prediction_id: UUID
    utterance_id: UUID
    reference_surface_text: str
    hypothesis_surface_text: str
    cer: float = Field(ge=0)
    dialect: DialectPreservationMetric
    automatic_failure_candidates: tuple[str, ...] = ()
    review_id: UUID
    review_revision_count: int = Field(ge=1)
    review_verdict: Literal["confirmed", "rejected", "uncertain"]
    confirmed_failure_types: tuple[str, ...] = ()
    review_notes: str | None = None


class HumanReviewedBaselineReport(StrictSchema):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    report_id: str
    source_report_id: str
    experiment_id: str
    benchmark_id: str
    benchmark_version: str
    created_at: datetime
    model: ModelDescriptor
    automatic_metrics: AggregateMetrics
    human_review: HumanReviewSummary
    cases: tuple[ReviewedEvaluationCase, ...]
    limitations: tuple[str, ...] = ()
    gate_decision: str


class ModelComparison(StrictSchema):
    benchmark_id: str
    benchmark_version: str
    reports: tuple[BaselineReport, ...]


class ErrorExportRecord(StrictSchema):
    exported_at: datetime = Field(default_factory=utc_now)
    failure_taxonomy: tuple[str, ...]
    case: EvaluationCaseResult
