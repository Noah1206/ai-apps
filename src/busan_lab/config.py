"""Runtime configuration with conservative local defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LabSettings:
    data_root: Path
    max_upload_bytes: int = 100 * 1024 * 1024
    min_duration_ms: float = 250
    max_duration_ms: float = 120_000
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"

    @classmethod
    def from_environment(cls, data_root: Path | None = None) -> LabSettings:
        configured = data_root or Path(os.getenv("BUSAN_LAB_DATA_DIR", "data/lab"))
        return cls(data_root=configured.expanduser().resolve())
