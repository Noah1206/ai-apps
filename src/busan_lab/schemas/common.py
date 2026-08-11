"""Shared schema primitives."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION: Final[Literal["1.0.0"]] = "1.0.0"


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


class StrictSchema(BaseModel):
    """Base contract that rejects accidental fields and in-place mutation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class ReviewStatus(StrEnum):
    CANDIDATE = "candidate"
    HUMAN_REVIEWED = "human_reviewed"
    APPROVED = "approved"
    DEPRECATED = "deprecated"


class DatasetSplit(StrEnum):
    UNASSIGNED = "unassigned"
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class ConsentRecord(StrictSchema):
    """Consent is explicit and separates storage, research, and training use."""

    storage_allowed: bool
    research_use_allowed: bool
    model_training_allowed: bool = False
    policy_version: str = Field(min_length=1, default="lab-v0")
    captured_at: datetime = Field(default_factory=utc_now)
