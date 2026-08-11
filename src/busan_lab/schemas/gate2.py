"""Gate 2 closure contracts for integrity, review, and held-out evaluation."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from busan_lab.schemas.common import SCHEMA_VERSION, StrictSchema, utc_now

Sha256 = str


class BenchmarkSerialization(StrictSchema):
    path: str = Field(min_length=1)
    raw_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    utterance_count: int = Field(ge=1)
    raw_dialect_expression_count: int = Field(ge=0)
    semantic_dialect_expression_count: int = Field(ge=0)
    annotation_structure: Literal["legacy_quoted_compound", "expanded_labels"]
    used_by: tuple[str, ...] = ()
    related_artifacts: tuple[str, ...] = ()


class BenchmarkIntegrityAudit(StrictSchema):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    audit_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    benchmark_id: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    decision: Literal["audited_semantic_equivalence_keep_version"]
    canonical_artifact_path: str = Field(min_length=1)
    canonical_package_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_manifest_raw_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_semantic_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    semantically_equivalent: Literal[True] = True
    serializations: tuple[BenchmarkSerialization, ...] = Field(min_length=2)
    provenance_note: str = Field(min_length=1)

    @model_validator(mode="after")
    def semantic_hashes_must_match(self) -> Self:
        if any(
            item.semantic_sha256 != self.canonical_semantic_sha256
            for item in self.serializations
        ):
            raise ValueError("all audited serializations must have the canonical semantic hash")
        return self


class ReproducibilityArtifact(StrictSchema):
    artifact_id: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    required: bool = True
    expected_sha256: Sha256 | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_gpu_path: str | None = None
    purpose: str = Field(min_length=1)

    @field_validator("relative_path")
    @classmethod
    def relative_path_must_be_safe(cls, value: str) -> str:
        from pathlib import PurePosixPath

        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or "\\" in value:
            raise ValueError("artifact relative_path must be a safe POSIX relative path")
        return value


class ReproducibilitySpec(StrictSchema):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    task_id: Literal["TASK-005"] = "TASK-005"
    experiment_id: str = Field(min_length=1)
    base_model_id: str = Field(min_length=1)
    base_model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    nemo_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    train_dataset_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    validation_dataset_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_package_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_selection_criterion: str = Field(min_length=1)
    artifacts: tuple[ReproducibilityArtifact, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def artifact_ids_and_paths_must_be_unique(self) -> Self:
        ids = [item.artifact_id for item in self.artifacts]
        paths = [item.relative_path for item in self.artifacts]
        if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
            raise ValueError("artifact IDs and paths must be unique")
        return self


ABPreference = Literal["A", "B", "tie", "uncertain"]
ABFinding = Literal["present", "absent", "uncertain"]


class BlindABReviewItem(StrictSchema):
    item_id: str = Field(min_length=1)
    utterance_id: str = Field(min_length=1)
    transcript_preference: ABPreference | None = None
    dialect_preservation_preference: ABPreference | None = None
    meaning_fidelity_preference: ABPreference | None = None
    meaning_distortion_a: ABFinding | None = None
    meaning_distortion_b: ABFinding | None = None
    overcorrection_a: ABFinding | None = None
    overcorrection_b: ABFinding | None = None
    notes: str | None = Field(default=None, max_length=4000)


class BlindABReviewResult(StrictSchema):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    review_id: str = Field(min_length=1)
    benchmark_id: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    benchmark_semantic_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer_id: str = Field(min_length=1)
    status: Literal["in_progress", "complete"] = "in_progress"
    items: tuple[BlindABReviewItem, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def completed_reviews_need_every_dimension(self) -> Self:
        ids = [item.item_id for item in self.items]
        utterance_ids = [item.utterance_id for item in self.items]
        if len(ids) != len(set(ids)) or len(utterance_ids) != len(set(utterance_ids)):
            raise ValueError("review item and utterance IDs must be unique")
        if self.status == "complete":
            required_fields = (
                "transcript_preference",
                "dialect_preservation_preference",
                "meaning_fidelity_preference",
                "meaning_distortion_a",
                "meaning_distortion_b",
                "overcorrection_a",
                "overcorrection_b",
            )
            for item in self.items:
                if any(getattr(item, field) is None for field in required_fields):
                    raise ValueError(f"completed review item is incomplete: {item.item_id}")
        return self


class Gate2EvaluationEntry(StrictSchema):
    utterance_id: str = Field(min_length=1)
    speaker_id: str = Field(min_length=1)
    source_recording_id: str = Field(min_length=1)
    region: str = Field(min_length=1)
    audio_filepath: str = Field(min_length=1)
    audio_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    audio_lineage_sha256s: tuple[Sha256, ...] = ()
    duration_seconds: float = Field(gt=0)
    surface_text: str = Field(min_length=1)
    dialect_expressions: tuple[str, ...] = ()
    label_status: Literal["human_reviewed", "approved"]

    @field_validator("audio_filepath")
    @classmethod
    def audio_path_must_be_safe(cls, value: str) -> str:
        from pathlib import PurePosixPath

        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or "\\" in value:
            raise ValueError("audio_filepath must be a safe POSIX relative path")
        return value


class Gate2EvaluationManifest(StrictSchema):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    dataset_kind: Literal["independent_busan_test", "standard_korean_regression"]
    created_at: datetime = Field(default_factory=utc_now)
    frozen: Literal[True] = True
    target_language: Literal["ko-KR"] = "ko-KR"
    training_allowed: Literal[False] = False
    checkpoint_selection_allowed: Literal[False] = False
    source_name: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    license_or_access_policy: str = Field(min_length=1)
    entries: tuple[Gate2EvaluationEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def identities_and_audio_must_be_unique(self) -> Self:
        fields = {
            "utterance_id": [entry.utterance_id for entry in self.entries],
            "audio_sha256": [entry.audio_sha256 for entry in self.entries],
        }
        for name, values in fields.items():
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {name}")
        return self


class EvaluationExclusionRegistry(StrictSchema):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    created_at: datetime = Field(default_factory=utc_now)
    source_artifacts: tuple[str, ...] = Field(min_length=1)
    speaker_ids: tuple[str, ...]
    utterance_ids: tuple[str, ...]
    source_recording_ids: tuple[str, ...]
    audio_sha256s: tuple[Sha256, ...]
    normalized_surface_texts: tuple[str, ...]


class Gate2Criteria(StrictSchema):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    status: Literal["proposed_thresholds"] = "proposed_thresholds"
    pilot_max_cer: float = Field(ge=0)
    pilot_min_dialect_preservation: float = Field(ge=0, le=1)
    human_min_fine_tuned_preference_rate: float = Field(ge=0, le=1)
    independent_min_utterances: int = Field(ge=1)
    independent_min_speakers: int = Field(ge=2)
    independent_min_relative_cer_improvement: float = Field(ge=0, le=1)
    independent_min_dialect_preservation_delta: float = Field(ge=0, le=1)
    standard_min_utterances: int = Field(ge=1)
    standard_min_speakers: int = Field(ge=2)
    standard_max_relative_cer_regression: float = Field(ge=0)
    standard_max_absolute_cer_regression: float = Field(ge=0)
    empty_output_increase_allowed: int = Field(ge=0)


class Gate2Evidence(StrictSchema):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    benchmark_integrity_passed: bool
    reproducibility_passed: bool
    pilot_pretrained_cer: float = Field(ge=0)
    pilot_fine_tuned_cer: float = Field(ge=0)
    pilot_pretrained_dialect_preservation: float = Field(ge=0, le=1)
    pilot_fine_tuned_dialect_preservation: float = Field(ge=0, le=1)
    pilot_pretrained_empty_outputs: int = Field(ge=0)
    pilot_fine_tuned_empty_outputs: int = Field(ge=0)
    human_review_complete: bool
    human_fine_tuned_preference_rate: float | None = Field(default=None, ge=0, le=1)
    independent_test_complete: bool
    independent_utterances: int = Field(ge=0)
    independent_speakers: int = Field(ge=0)
    independent_pretrained_cer: float | None = Field(default=None, ge=0)
    independent_fine_tuned_cer: float | None = Field(default=None, ge=0)
    independent_pretrained_dialect_preservation: float | None = Field(
        default=None, ge=0, le=1
    )
    independent_fine_tuned_dialect_preservation: float | None = Field(
        default=None, ge=0, le=1
    )
    independent_pretrained_empty_outputs: int | None = Field(default=None, ge=0)
    independent_fine_tuned_empty_outputs: int | None = Field(default=None, ge=0)
    standard_regression_complete: bool
    standard_utterances: int = Field(ge=0)
    standard_speakers: int = Field(ge=0)
    standard_pretrained_cer: float | None = Field(default=None, ge=0)
    standard_fine_tuned_cer: float | None = Field(default=None, ge=0)
    standard_pretrained_empty_outputs: int | None = Field(default=None, ge=0)
    standard_fine_tuned_empty_outputs: int | None = Field(default=None, ge=0)


class Gate2Assessment(StrictSchema):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    status: Literal["PASS", "CONDITIONAL PASS", "FAIL"]
    passed_checks: tuple[str, ...]
    failed_checks: tuple[str, ...]
    pending_checks: tuple[str, ...]
    note: str = Field(min_length=1)
