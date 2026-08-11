"""Keep all project Python commands inside the pinned uv environment."""

from __future__ import annotations

import platform
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

from busan_lab.schemas.environment import EnvironmentCheck, EnvironmentReport


def find_project_root(start: Path | None = None) -> Path:
    """Find the nearest Busan Speech Research Lab pyproject."""

    candidate = (start or Path.cwd()).resolve()
    for directory in (candidate, *candidate.parents):
        pyproject = directory / "pyproject.toml"
        if pyproject.is_file() and 'name = "busan-speech-research-lab"' in pyproject.read_text(
            encoding="utf-8"
        ):
            return directory
    raise FileNotFoundError("Busan Speech Research Lab project root was not found")


def inspect_project_environment(
    project_root: Path,
    *,
    python_prefix: Path | None = None,
    python_version: str | None = None,
    executable_lookup: Callable[[str], str | None] = shutil.which,
) -> EnvironmentReport:
    """Return evidence that commands use this project's one pinned environment."""

    root = project_root.resolve()
    expected_environment = (root / ".venv").resolve()
    active_environment = (python_prefix or Path(sys.prefix)).resolve()
    active_version = python_version or platform.python_version()
    version_file = root / ".python-version"
    pinned_version = (
        version_file.read_text(encoding="utf-8").strip() if version_file.is_file() else ""
    )
    ffmpeg_path = executable_lookup("ffmpeg")
    ffprobe_path = executable_lookup("ffprobe")

    checks = (
        EnvironmentCheck(
            name="project_venv",
            passed=active_environment == expected_environment,
            detail=f"active={active_environment}; expected={expected_environment}",
        ),
        EnvironmentCheck(
            name="python_version",
            passed=bool(pinned_version) and active_version == pinned_version,
            detail=f"active={active_version}; pinned={pinned_version or 'missing'}",
        ),
        EnvironmentCheck(
            name="uv_lock",
            passed=(root / "uv.lock").is_file(),
            detail=str(root / "uv.lock"),
        ),
        EnvironmentCheck(
            name="ffmpeg",
            passed=ffmpeg_path is not None,
            detail=ffmpeg_path or "not found",
        ),
        EnvironmentCheck(
            name="ffprobe",
            passed=ffprobe_path is not None,
            detail=ffprobe_path or "not found",
        ),
    )
    return EnvironmentReport(
        project_root=str(root),
        expected_environment=str(expected_environment),
        active_environment=str(active_environment),
        python_version=active_version,
        checks=checks,
        passed=all(check.passed for check in checks),
    )
