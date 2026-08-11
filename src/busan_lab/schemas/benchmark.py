"""Fixed benchmark manifest contracts and leakage guards."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from busan_lab.schemas.common import (
    SCHEMA_VERSION,
    DatasetSplit,
    StrictSchema,
    utc_now,
)
from busan_lab.schemas.utterance import (
    DialectExpressionLabel,
    expand_dialect_expression_labels,
)


class BenchmarkEntry(StrictSchema):
    utterance_id: UUID
    speaker_id: str = Field(min_length=1)
    split: DatasetSplit
    original_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    derived_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    derived_audio_path: str = Field(min_length=1)
    lineage_audio_sha256s: tuple[str, ...] = ()
    surface_text: str = Field(min_length=1)
    normalized_meaning: str | None = None
    dialect_expressions: tuple[DialectExpressionLabel, ...] = ()

    @field_validator("dialect_expressions", mode="after")
    @classmethod
    def expand_quoted_expressions(
        cls,
        labels: tuple[DialectExpressionLabel, ...],
    ) -> tuple[DialectExpressionLabel, ...]:
        return expand_dialect_expression_labels(labels)

    @model_validator(mode="after")
    def benchmark_entries_need_a_split(self) -> Self:
        if self.split is DatasetSplit.UNASSIGNED:
            raise ValueError("benchmark entries require train, validation, or test split")
        return self


class BenchmarkManifest(StrictSchema):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    benchmark_id: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    frozen: bool = True
    entries: tuple[BenchmarkEntry, ...]

    @model_validator(mode="after")
    def prevent_identity_and_lineage_leakage(self) -> Self:
        utterance_ids: set[UUID] = set()
        speaker_splits: dict[str, DatasetSplit] = {}
        audio_splits: dict[str, DatasetSplit] = {}
        for entry in self.entries:
            if entry.utterance_id in utterance_ids:
                raise ValueError(f"duplicate utterance_id: {entry.utterance_id}")
            utterance_ids.add(entry.utterance_id)

            prior_speaker_split = speaker_splits.setdefault(entry.speaker_id, entry.split)
            if prior_speaker_split is not entry.split:
                raise ValueError(
                    f"speaker {entry.speaker_id!r} crosses "
                    f"{prior_speaker_split.value}/{entry.split.value}"
                )

            for audio_hash in (
                entry.original_audio_sha256,
                entry.derived_audio_sha256,
                *entry.lineage_audio_sha256s,
            ):
                prior_audio_split = audio_splits.setdefault(audio_hash, entry.split)
                if prior_audio_split is not entry.split:
                    raise ValueError(
                        f"audio lineage {audio_hash} crosses "
                        f"{prior_audio_split.value}/{entry.split.value}"
                    )
        return self
