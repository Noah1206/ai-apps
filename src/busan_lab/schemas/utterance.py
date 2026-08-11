"""Utterance and linguistic ground-truth contracts."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import Field, field_validator

from busan_lab.schemas.audio import AudioBundle
from busan_lab.schemas.common import (
    SCHEMA_VERSION,
    ConsentRecord,
    DatasetSplit,
    ReviewStatus,
    StrictSchema,
    utc_now,
)


class DialectExpressionLabel(StrictSchema):
    """A linguistic label; generated labels remain candidates until reviewed."""

    surface_form: str = Field(min_length=1)
    normalized_forms: tuple[str, ...] = ()
    status: ReviewStatus = ReviewStatus.CANDIDATE
    reviewer_id: str | None = None
    notes: str | None = None


_QUOTED_LABEL_TERM = re.compile(r'["“]([^"”]+)["”]')


def _split_label_terms(value: str) -> tuple[str, ...]:
    matches = tuple(match.group(1).strip() for match in _QUOTED_LABEL_TERM.finditer(value))
    if matches:
        remainder = _QUOTED_LABEL_TERM.sub("", value)
        if not remainder.replace(",", "").replace("\uff0c", "").strip():
            return matches
    cleaned = value.strip().strip('"“”\'').strip()
    return (cleaned,) if cleaned else ()


def expand_dialect_expression_labels(
    labels: tuple[DialectExpressionLabel, ...],
) -> tuple[DialectExpressionLabel, ...]:
    """Turn user-quoted expression lists into individual labels."""

    expanded: list[DialectExpressionLabel] = []
    for label in labels:
        surfaces = _split_label_terms(label.surface_form)
        normalized = tuple(
            term
            for value in label.normalized_forms
            for term in _split_label_terms(value)
        )
        if len(surfaces) > 1 and normalized and len(normalized) != len(surfaces):
            raise ValueError(
                f"Surface expressions ({len(surfaces)}) and normalized forms "
                f"({len(normalized)}) must have the same count."
            )
        for index, surface in enumerate(surfaces):
            forms = (normalized[index],) if len(surfaces) > 1 and normalized else normalized
            expanded.append(
                label.model_copy(
                    update={"surface_form": surface, "normalized_forms": forms}
                )
            )
    return tuple(expanded)


class LinguisticGroundTruth(StrictSchema):
    """Surface text is never replaced by normalized meaning."""

    surface_text: str = Field(min_length=1)
    normalized_meaning: str | None = None
    dialect_expressions: tuple[DialectExpressionLabel, ...] = ()
    label_status: ReviewStatus = ReviewStatus.CANDIDATE
    label_version: str = Field(min_length=1, default="label_v0")
    reviewer_id: str | None = None

    @field_validator("dialect_expressions", mode="after")
    @classmethod
    def expand_quoted_expressions(
        cls,
        labels: tuple[DialectExpressionLabel, ...],
    ) -> tuple[DialectExpressionLabel, ...]:
        return expand_dialect_expression_labels(labels)


class SpeakerContext(StrictSchema):
    speaker_id: str = Field(min_length=1)
    region: str = Field(min_length=1)
    age_group: str | None = None
    gender: str | None = None
    device: str = Field(min_length=1, default="unknown")
    environment: str = Field(min_length=1, default="unknown")


class UtteranceRecord(StrictSchema):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    utterance_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=utc_now)
    source: Literal["lab_upload", "import"] = "lab_upload"
    speaker: SpeakerContext
    dataset_split: DatasetSplit = DatasetSplit.UNASSIGNED
    consent: ConsentRecord
    audio: AudioBundle
    ground_truth: LinguisticGroundTruth


class LabelRevision(StrictSchema):
    """Append-only evidence for a correction to linguistic ground truth."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    revision_id: UUID = Field(default_factory=uuid4)
    utterance_id: UUID
    previous: LinguisticGroundTruth
    updated: LinguisticGroundTruth
    changed_by: str = Field(min_length=1, max_length=128)
    reason: str | None = Field(default=None, max_length=4000)
    created_at: datetime = Field(default_factory=utc_now)
