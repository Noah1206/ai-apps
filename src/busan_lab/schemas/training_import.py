"""Contracts for append-only TASK-004 recording imports."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import PurePath
from typing import Literal, Self
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from busan_lab.schemas.common import (
    SCHEMA_VERSION,
    ConsentRecord,
    DatasetSplit,
    ReviewStatus,
    StrictSchema,
    utc_now,
)


class TrainingRecordingPlanItem(StrictSchema):
    prompt_id: str = Field(pattern=r"^T004-S\d{3}$")
    source_filename: str = Field(min_length=1)
    source_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_surface_text: str = Field(min_length=1)

    @field_validator("source_filename")
    @classmethod
    def source_filename_must_be_a_basename(cls, value: str) -> str:
        if PurePath(value).name != value or "/" in value or "\\" in value:
            raise ValueError("source_filename must not contain a directory")
        return value


class TrainingRecordingImportPlan(StrictSchema):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    task_id: Literal["TASK-004"] = "TASK-004"
    import_id: str = Field(min_length=1, max_length=128)
    source_directory_name: str = Field(min_length=1)
    prompt_sheet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    speaker_id: str = Field(min_length=1)
    region: str = Field(min_length=1)
    device: str = Field(min_length=1)
    recording_environment: str = Field(min_length=1)
    split: Literal[DatasetSplit.TRAIN] = DatasetSplit.TRAIN
    label_status: Literal[ReviewStatus.CANDIDATE] = ReviewStatus.CANDIDATE
    consent: ConsentRecord
    expected_recordings: int = Field(gt=0)
    items: tuple[TrainingRecordingPlanItem, ...] = ()
    benchmark_manifests_checked: tuple[str, ...]
    passed: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        if self.passed and len(self.items) != self.expected_recordings:
            raise ValueError("recording plan count must match expected_recordings")
        if self.passed == bool(self.errors):
            raise ValueError("plan passed must be true exactly when errors are empty")
        for attribute in ("prompt_id", "source_filename", "source_audio_sha256"):
            values = [getattr(item, attribute) for item in self.items]
            if len(values) != len(set(values)):
                raise ValueError(f"recording plan contains duplicate {attribute}")
        return self


class TrainingRecordingImportEntry(TrainingRecordingPlanItem):
    utterance_id: UUID
    duration_ms: float = Field(gt=0)
    audio_quality_passed: bool
    audio_quality_warnings: tuple[str, ...] = ()


class TrainingRecordingImportManifest(StrictSchema):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    task_id: Literal["TASK-004"] = "TASK-004"
    import_id: str = Field(min_length=1, max_length=128)
    created_at: datetime = Field(default_factory=utc_now)
    source_directory_name: str = Field(min_length=1)
    prompt_sheet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    speaker_id: str = Field(min_length=1)
    region: str = Field(min_length=1)
    device: str = Field(min_length=1)
    recording_environment: str = Field(min_length=1)
    split: Literal[DatasetSplit.TRAIN] = DatasetSplit.TRAIN
    label_status: Literal[ReviewStatus.CANDIDATE] = ReviewStatus.CANDIDATE
    consent: ConsentRecord
    entries: tuple[TrainingRecordingImportEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def entries_must_be_unique_and_quality_checked(self) -> Self:
        for attribute in (
            "prompt_id",
            "source_filename",
            "source_audio_sha256",
            "utterance_id",
        ):
            values = [getattr(entry, attribute) for entry in self.entries]
            if len(values) != len(set(values)):
                raise ValueError(f"recording import contains duplicate {attribute}")
        if any(not entry.audio_quality_passed for entry in self.entries):
            raise ValueError("recording import cannot complete with failed audio quality")
        return self


class TrainingRecordingReviewDecision(StrEnum):
    """Human decisions available in the TASK-004 recording review queue."""

    APPROVE = "approve"
    RERECORD = "rerecord"


class TrainingRecordingImportSummary(StrictSchema):
    """Compact import metadata for selecting one review queue."""

    task_id: Literal["TASK-004"] = "TASK-004"
    import_id: str = Field(min_length=1, max_length=128)
    created_at: datetime
    speaker_id: str = Field(min_length=1)
    entry_count: int = Field(gt=0)


class TrainingRecordingReviewItem(StrictSchema):
    """One imported recording joined with its latest editable label."""

    position: int = Field(gt=0)
    prompt_id: str = Field(pattern=r"^T004-S\d{3}$")
    utterance_id: UUID
    source_filename: str = Field(min_length=1)
    candidate_surface_text: str = Field(min_length=1)
    surface_text: str = Field(min_length=1)
    duration_ms: float = Field(gt=0)
    audio_quality_passed: bool
    audio_quality_warnings: tuple[str, ...] = ()
    label_status: ReviewStatus
    label_version: str = Field(min_length=1)


class TrainingRecordingReviewQueue(StrictSchema):
    """Ordered review state derived from an immutable recording import."""

    task_id: Literal["TASK-004"] = "TASK-004"
    import_id: str = Field(min_length=1, max_length=128)
    speaker_id: str = Field(min_length=1)
    total_count: int = Field(gt=0)
    reviewed_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    human_reviewed_count: int = Field(ge=0)
    approved_count: int = Field(ge=0)
    rerecord_count: int = Field(ge=0)
    items: tuple[TrainingRecordingReviewItem, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def counts_must_match_items(self) -> Self:
        status_counts = {
            status: sum(item.label_status is status for item in self.items)
            for status in ReviewStatus
        }
        if self.total_count != len(self.items):
            raise ValueError("review queue total_count must match items")
        if self.candidate_count != status_counts[ReviewStatus.CANDIDATE]:
            raise ValueError("review queue candidate_count does not match items")
        if self.human_reviewed_count != status_counts[ReviewStatus.HUMAN_REVIEWED]:
            raise ValueError("review queue human_reviewed_count does not match items")
        if self.approved_count != status_counts[ReviewStatus.APPROVED]:
            raise ValueError("review queue approved_count does not match items")
        if self.rerecord_count != status_counts[ReviewStatus.DEPRECATED]:
            raise ValueError("review queue rerecord_count does not match items")
        if self.reviewed_count != self.total_count - self.candidate_count:
            raise ValueError("review queue reviewed_count does not match items")
        return self


class TrainingRecordingReviewRequest(StrictSchema):
    """A human approval or re-recording decision for one imported recording."""

    reviewer_id: str = Field(min_length=1, max_length=128)
    decision: TrainingRecordingReviewDecision
    notes: str | None = Field(default=None, max_length=4000)
