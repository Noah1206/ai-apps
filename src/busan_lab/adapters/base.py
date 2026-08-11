"""Replaceable contract for a pretrained Korean Surface ASR baseline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from busan_lab.schemas.asr import ModelDescriptor, SurfaceASRResult


class SurfaceASRAdapter(ABC):
    """Keep any model/runtime behind one Surface ASR boundary."""

    @property
    @abstractmethod
    def model(self) -> ModelDescriptor:
        """Describe the exact model/version/checkpoint under evaluation."""

    @property
    @abstractmethod
    def experiment_id(self) -> str:
        """Identify the reproducible model run that produced these predictions."""

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        *,
        utterance_id: str,
        audio_sha256: str,
    ) -> SurfaceASRResult:
        """Return a surface-form hypothesis without normalizing dialect text."""
