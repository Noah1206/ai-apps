"""Project-local Python environment diagnostics."""

from __future__ import annotations

from pydantic import Field

from busan_lab.schemas.common import StrictSchema


class EnvironmentCheck(StrictSchema):
    name: str = Field(min_length=1)
    passed: bool
    detail: str = Field(min_length=1)


class EnvironmentReport(StrictSchema):
    project_root: str = Field(min_length=1)
    expected_environment: str = Field(min_length=1)
    active_environment: str = Field(min_length=1)
    python_version: str = Field(min_length=1)
    checks: tuple[EnvironmentCheck, ...]
    passed: bool
