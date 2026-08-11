"""Versioned contracts for calibrated Surface ASR evaluation."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import Field

from busan_lab.schemas.asr import ModelDescriptor
from busan_lab.schemas.common import ReviewStatus, StrictSchema, utc_now
from busan_lab.schemas.evaluation import AggregateMetrics, DialectExpressionResult


class ObservedError(StrEnum):
    NO_ERROR = "NO_ERROR"
    DIALECT_EXPRESSION_LOST = "DIALECT_EXPRESSION_LOST"
    DIALECT_TO_STANDARD = "DIALECT_TO_STANDARD"
    PHONETIC_SUBSTITUTION = "PHONETIC_SUBSTITUTION"
    ENDING_SUBSTITUTION = "ENDING_SUBSTITUTION"
    WORD_OR_SYLLABLE_OMISSION = "WORD_OR_SYLLABLE_OMISSION"
    WORD_OR_SYLLABLE_INSERTION = "WORD_OR_SYLLABLE_INSERTION"
    WORD_BOUNDARY_ERROR = "WORD_BOUNDARY_ERROR"
    LABEL_ERROR = "LABEL_ERROR"
    AUDIO_QUALITY_ERROR = "AUDIO_QUALITY_ERROR"
    AMBIGUOUS_VARIANT = "AMBIGUOUS_VARIANT"
    UNKNOWN = "UNKNOWN"


class SuspectedCause(StrEnum):
    ACOUSTIC_CONFUSION = "ACOUSTIC_CONFUSION"
    STANDARD_KOREAN_MODEL_BIAS = "STANDARD_KOREAN_MODEL_BIAS"
    DECODER_BIAS = "DECODER_BIAS"
    TOKENIZER_LIMITATION = "TOKENIZER_LIMITATION"
    AUDIO_QUALITY = "AUDIO_QUALITY"
    LABEL_ERROR = "LABEL_ERROR"
    UNKNOWN = "UNKNOWN"


class DialectMatchKind(StrEnum):
    EXACT_SURFACE = "EXACT_SURFACE"
    ACCEPTABLE_VARIANT = "ACCEPTABLE_VARIANT"
    STANDARD_EQUIVALENT = "STANDARD_EQUIVALENT"
    MISSING = "MISSING"


class HumanComparisonOutcome(StrEnum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    MISSED = "MISSED"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    NO_ERROR_AGREEMENT = "NO_ERROR_AGREEMENT"
    UNCERTAIN = "UNCERTAIN"


class CalibratedForm(StrictSchema):
    form: str = Field(min_length=1)
    status: ReviewStatus = ReviewStatus.CANDIDATE
    evidence_review_ids: tuple[UUID, ...] = ()
    notes: str | None = None


class DialectCalibrationRule(StrictSchema):
    surface_form: str = Field(min_length=1)
    acceptable_variants: tuple[CalibratedForm, ...] = ()
    standard_equivalents: tuple[CalibratedForm, ...] = ()


class EvaluationCalibrationProfile(StrictSchema):
    schema_version: Literal["1.0.0"] = "1.0.0"
    revision_id: str = Field(min_length=1, max_length=128)
    benchmark_id: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    rules: tuple[DialectCalibrationRule, ...] = ()
    notes: tuple[str, ...] = ()


class ErrorObservation(StrictSchema):
    observed_error: ObservedError
    suspected_cause: SuspectedCause = SuspectedCause.UNKNOWN
    surface_form: str | None = None
    matched_form: str | None = None
    mapping_status: ReviewStatus | None = None
    evidence: str | None = None


class PredictionDiagnostics(StrictSchema):
    """Read-only calibrated observations for one immutable prediction."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    prediction_id: UUID
    calibration_revision: str | None = None
    observations: tuple[ErrorObservation, ...]
    automatic_failure_candidates: tuple[str, ...]


class CalibratedDialectResult(StrictSchema):
    surface_form: str
    match_kind: DialectMatchKind
    matched_form: str | None = None
    mapping_status: ReviewStatus | None = None
    observations: tuple[ErrorObservation, ...]


class CalibratedAggregateMetrics(StrictSchema):
    reference_expression_count: int = Field(ge=0)
    evaluated_expression_count: int = Field(ge=0)
    ambiguous_expression_count: int = Field(ge=0)
    exact_preserved_count: int = Field(ge=0)
    acceptable_variant_count: int = Field(ge=0)
    standard_equivalent_candidate_count: int = Field(ge=0)
    missing_expression_count: int = Field(ge=0)
    dialect_preservation_rate: float = Field(ge=0, le=1)
    context_overcorrection_candidate_rate: float = Field(ge=0, le=1)


class HumanComparisonSummary(StrictSchema):
    automatic_human_match_count: int = Field(ge=0)
    automatic_human_mismatch_count: int = Field(ge=0)
    automatic_missed_count: int = Field(ge=0)
    automatic_false_positive_count: int = Field(ge=0)
    no_error_agreement_count: int = Field(ge=0)
    uncertain_count: int = Field(ge=0)


class CalibratedEvaluationCase(StrictSchema):
    prediction_id: UUID
    utterance_id: UUID
    reference_surface_text: str
    hypothesis_surface_text: str
    legacy_dialect_results: tuple[DialectExpressionResult, ...]
    legacy_automatic_failure_candidates: tuple[str, ...] = ()
    calibrated_dialect_results: tuple[CalibratedDialectResult, ...]
    observations: tuple[ErrorObservation, ...]
    automatic_failure_candidates: tuple[str, ...]
    latest_review_id: UUID
    review_revision_count: int = Field(ge=1)
    human_verdict: Literal["confirmed", "rejected", "uncertain"]
    human_confirmed_failure_types: tuple[str, ...] = ()
    human_review_notes: str | None = None
    comparison_outcome: HumanComparisonOutcome


class EvaluationCalibrationReport(StrictSchema):
    schema_version: Literal["1.0.0"] = "1.0.0"
    report_id: str
    evaluation_revision: str
    source_report_id: str
    source_reviewed_report_id: str | None = None
    experiment_id: str
    benchmark_id: str
    benchmark_version: str
    created_at: datetime
    model: ModelDescriptor
    legacy_automatic_metrics: AggregateMetrics
    calibrated_metrics: CalibratedAggregateMetrics
    human_comparison: HumanComparisonSummary
    cases: tuple[CalibratedEvaluationCase, ...]
    limitations: tuple[str, ...] = ()
    task_003b_prediction_contract_compatible: bool
