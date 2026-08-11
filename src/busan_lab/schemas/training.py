"""TASK-004 contracts for leakage-safe Surface ASR training datasets."""

from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath
from typing import Literal, Self
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from busan_lab.schemas.common import (
    SCHEMA_VERSION,
    DatasetSplit,
    ReviewStatus,
    StrictSchema,
    utc_now,
)
from busan_lab.schemas.utterance import DialectExpressionLabel


class TrainingDatasetEntry(StrictSchema):
    """One approved, training-consented Surface ASR example."""

    utterance_id: UUID
    speaker_id: str = Field(min_length=1)
    split: DatasetSplit
    region: str = Field(min_length=1)
    recording_environment: str = Field(min_length=1)
    original_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    asr_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    asr_audio_path: str = Field(min_length=1)
    audio_lineage_sha256s: tuple[str, ...] = Field(min_length=2)
    duration_ms: float = Field(gt=0)
    surface_text: str = Field(min_length=1)
    label_status: ReviewStatus
    label_version: str = Field(min_length=1)
    dialect_expressions: tuple[DialectExpressionLabel, ...] = ()
    audio_quality_passed: Literal[True] = True
    model_training_allowed: Literal[True] = True
    consent_policy_version: str = Field(min_length=1)

    @field_validator("asr_audio_path")
    @classmethod
    def audio_path_must_be_safe(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or "\\" in value:
            raise ValueError("asr_audio_path must be a safe POSIX relative path")
        return value

    @field_validator("audio_lineage_sha256s")
    @classmethod
    def lineage_hashes_must_be_valid(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(
            len(audio_hash) != 64
            or any(character not in "0123456789abcdef" for character in audio_hash)
            for audio_hash in value
        ):
            raise ValueError("audio lineage values must be lowercase SHA-256 hashes")
        if len(value) != len(set(value)):
            raise ValueError("audio lineage values must be unique")
        return value

    @model_validator(mode="after")
    def enforce_training_eligibility(self) -> Self:
        if self.split not in {DatasetSplit.TRAIN, DatasetSplit.VALIDATION}:
            raise ValueError("training entries require train or validation split")
        if self.label_status not in {
            ReviewStatus.HUMAN_REVIEWED,
            ReviewStatus.APPROVED,
        }:
            raise ValueError("training entries require a human-reviewed or approved label")
        if self.original_audio_sha256 not in self.audio_lineage_sha256s:
            raise ValueError("original audio must be included in audio lineage")
        if self.asr_audio_sha256 not in self.audio_lineage_sha256s:
            raise ValueError("ASR audio must be included in audio lineage")
        return self


class TrainingDatasetManifest(StrictSchema):
    """Frozen train/validation membership with strict split isolation."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    task_id: Literal["TASK-004"] = "TASK-004"
    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    frozen: Literal[True] = True
    benchmark_exclusion_policy: Literal[
        "utterance_speaker_audio_lineage_and_exact_surface"
    ] = "utterance_speaker_audio_lineage_and_exact_surface"
    entries: tuple[TrainingDatasetEntry, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def prevent_training_split_leakage(self) -> Self:
        utterance_ids: set[UUID] = set()
        speakers: dict[str, DatasetSplit] = {}
        audio_hashes: dict[str, UUID] = {}
        splits: set[DatasetSplit] = set()
        for entry in self.entries:
            if entry.utterance_id in utterance_ids:
                raise ValueError(f"duplicate utterance_id: {entry.utterance_id}")
            utterance_ids.add(entry.utterance_id)
            splits.add(entry.split)

            prior_split = speakers.setdefault(entry.speaker_id, entry.split)
            if prior_split is not entry.split:
                raise ValueError(
                    f"speaker {entry.speaker_id!r} crosses "
                    f"{prior_split.value}/{entry.split.value}"
                )
            for audio_hash in entry.audio_lineage_sha256s:
                prior_utterance = audio_hashes.setdefault(audio_hash, entry.utterance_id)
                if prior_utterance != entry.utterance_id:
                    raise ValueError(
                        f"duplicate audio lineage {audio_hash} appears in "
                        f"{prior_utterance}/{entry.utterance_id}"
                    )
        if splits != {DatasetSplit.TRAIN, DatasetSplit.VALIDATION}:
            raise ValueError("training dataset requires both train and validation entries")
        return self


class TrainingSplitAssignments(StrictSchema):
    """User-reviewed membership input for one immutable training manifest."""

    train_utterance_ids: tuple[UUID, ...] = Field(min_length=1)
    validation_utterance_ids: tuple[UUID, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def assignments_must_be_unique(self) -> Self:
        combined = (*self.train_utterance_ids, *self.validation_utterance_ids)
        if len(combined) != len(set(combined)):
            raise ValueError("training split assignments contain duplicate utterance IDs")
        return self


class DuplicateSurfaceTextGroup(StrictSchema):
    normalized_surface_text: str = Field(min_length=1)
    utterance_ids: tuple[UUID, ...] = Field(min_length=2)
    splits: tuple[DatasetSplit, ...] = Field(min_length=1)


class TrainingDatasetValidationReport(StrictSchema):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    task_id: Literal["TASK-004"] = "TASK-004"
    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    passed: bool
    benchmark_manifests_checked: tuple[str, ...]
    train_utterance_count: int = Field(ge=0)
    validation_utterance_count: int = Field(ge=0)
    train_speaker_count: int = Field(ge=0)
    validation_speaker_count: int = Field(ge=0)
    total_duration_hours: float = Field(ge=0)
    duplicate_surface_text_groups: tuple[DuplicateSurfaceTextGroup, ...] = ()
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def passed_must_match_errors(self) -> Self:
        if self.passed == bool(self.errors):
            raise ValueError("validation passed must be true exactly when errors are empty")
        return self


class TrainingExportRecord(StrictSchema):
    """Model-neutral JSONL row; text is always the approved Surface transcript."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    utterance_id: UUID
    split: DatasetSplit
    audio_filepath: str = Field(min_length=1)
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    duration_seconds: float = Field(gt=0)
    text: str = Field(min_length=1)
    speaker_id: str = Field(min_length=1)
    dialect_expressions: tuple[str, ...] = ()

    @field_validator("audio_filepath")
    @classmethod
    def export_path_must_be_safe(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or "\\" in value:
            raise ValueError("audio_filepath must be a safe POSIX relative path")
        return value

    @model_validator(mode="after")
    def export_split_must_be_training_split(self) -> Self:
        if self.split not in {DatasetSplit.TRAIN, DatasetSplit.VALIDATION}:
            raise ValueError("training export rows require train or validation split")
        return self
