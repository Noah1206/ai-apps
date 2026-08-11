"""Audio metadata, lineage, quality, and visualization contracts."""

from __future__ import annotations

from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from busan_lab.schemas.common import SCHEMA_VERSION, StrictSchema


class AudioRole(StrEnum):
    """Stable roles in the master-audio lineage.

    ``ORIGINAL`` and ``DERIVED_16K_MONO`` remain readable for records created
    by Audio Lab v0.1. New records use a canonical 48 kHz master and
    purpose-specific derivatives.
    """

    ORIGINAL = "original"
    MASTER_48K_MONO = "master_48k_mono"
    MASTER_48K_STEREO = "master_48k_stereo"
    ASR_16K_MONO = "asr_16k_mono"
    PRONUNCIATION_24K_MONO = "pronunciation_24k_mono"
    TTS_48K_MONO = "tts_48k_mono"
    DERIVED_16K_MONO = "derived_16k_mono"


class AudioAsset(StrictSchema):
    asset_id: str = Field(min_length=8)
    role: AudioRole
    relative_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_size: int = Field(gt=0)
    container: str = Field(min_length=1)
    codec: str = Field(min_length=1)
    sample_rate_hz: int = Field(gt=0)
    channels: int = Field(gt=0)
    sample_width_bits: int | None = Field(default=None, gt=0)
    duration_ms: float = Field(gt=0)
    parent_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("relative_path")
    @classmethod
    def relative_path_must_be_safe(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("relative_path must remain inside the configured data root")
        return value


class AudioTransformation(StrictSchema):
    name: str = Field(min_length=1)
    target_role: AudioRole = AudioRole.DERIVED_16K_MONO
    tool: Literal["ffmpeg"] = "ffmpeg"
    tool_version: str = Field(min_length=1)
    command_fingerprint: str = Field(min_length=8)


class AudioQualityReport(StrictSchema):
    passed: bool
    duration_ms: float = Field(gt=0)
    peak_amplitude: float = Field(ge=0, le=1.1)
    rms_dbfs: float
    clipping_ratio: float = Field(ge=0, le=1)
    silence_ratio: float = Field(ge=0, le=1)
    dc_offset: float
    warnings: tuple[str, ...] = ()


class AudioBundle(StrictSchema):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    audio_contract_version: Literal["1.1.0"] = "1.1.0"
    original: AudioAsset
    derived: AudioAsset
    transformation: AudioTransformation
    quality: AudioQualityReport
    master: AudioAsset | None = None
    derivatives: tuple[AudioAsset, ...] = ()
    transformations: tuple[AudioTransformation, ...] = ()

    @model_validator(mode="after")
    def validate_lineage(self) -> Self:
        if self.original.role is not AudioRole.ORIGINAL:
            raise ValueError("original asset must have role=original")

        if self.master is None:
            return self._validate_legacy_lineage()

        if self.master.role not in {
            AudioRole.MASTER_48K_MONO,
            AudioRole.MASTER_48K_STEREO,
        }:
            raise ValueError("master asset must have a 48kHz master role")
        expected_master_channels = 1 if self.master.role is AudioRole.MASTER_48K_MONO else 2
        if (
            self.master.sample_rate_hz != 48_000
            or self.master.channels != expected_master_channels
            or self.master.codec != "pcm_s24le"
        ):
            raise ValueError("master audio must be 48kHz mono/stereo PCM24")
        if self.master.parent_sha256 != self.original.sha256:
            raise ValueError("master parent_sha256 must reference the uploaded original")

        assets_by_role = {asset.role: asset for asset in self.derivatives}
        if len(assets_by_role) != len(self.derivatives):
            raise ValueError("purpose-specific derivative roles must be unique")
        required_roles = {
            AudioRole.ASR_16K_MONO,
            AudioRole.PRONUNCIATION_24K_MONO,
            AudioRole.TTS_48K_MONO,
        }
        if set(assets_by_role) != required_roles:
            raise ValueError("ASR, pronunciation, and TTS derivatives are all required")
        for asset in self.derivatives:
            if asset.parent_sha256 != self.master.sha256:
                raise ValueError("every purpose-specific derivative must reference the master")

        asr_asset = assets_by_role[AudioRole.ASR_16K_MONO]
        if self.derived.sha256 != asr_asset.sha256:
            raise ValueError("derived compatibility alias must reference the ASR derivative")
        self._validate_asset_contract(
            asr_asset,
            sample_rate_hz=16_000,
            channels=1,
            codec="pcm_s16le",
        )
        self._validate_asset_contract(
            assets_by_role[AudioRole.PRONUNCIATION_24K_MONO],
            sample_rate_hz=24_000,
            channels=1,
            codec="pcm_s16le",
        )
        self._validate_asset_contract(
            assets_by_role[AudioRole.TTS_48K_MONO],
            sample_rate_hz=48_000,
            channels=1,
            codec="pcm_s24le",
        )

        transformations_by_role = {
            transformation.target_role: transformation for transformation in self.transformations
        }
        expected_transformation_roles = required_roles | {self.master.role}
        if set(transformations_by_role) != expected_transformation_roles:
            raise ValueError("each master/derivative asset requires a transformation record")
        if self.transformation.target_role is not AudioRole.ASR_16K_MONO:
            raise ValueError("transformation compatibility alias must describe the ASR derivative")
        return self

    def _validate_legacy_lineage(self) -> Self:
        if self.derived.role is not AudioRole.DERIVED_16K_MONO:
            raise ValueError("legacy derived asset must have role=derived_16k_mono")
        if self.derived.parent_sha256 != self.original.sha256:
            raise ValueError("legacy derived parent_sha256 must reference the original")
        if (
            self.derived.sample_rate_hz != 16_000
            or self.derived.channels != 1
            or self.derived.codec != "pcm_s16le"
        ):
            raise ValueError("legacy derived audio must be 16kHz mono PCM16")
        return self

    @staticmethod
    def _validate_asset_contract(
        asset: AudioAsset,
        *,
        sample_rate_hz: int,
        channels: int,
        codec: str,
    ) -> None:
        if (
            asset.sample_rate_hz != sample_rate_hz
            or asset.channels != channels
            or asset.codec != codec
        ):
            raise ValueError(f"{asset.role.value} must be {sample_rate_hz}Hz/{channels}ch/{codec}")

    def asset_for_role(self, role: AudioRole) -> AudioAsset:
        """Resolve an asset without exposing storage implementation details."""

        if role is AudioRole.ORIGINAL:
            return self.original
        if self.master is not None and role is self.master.role:
            return self.master
        if role is AudioRole.DERIVED_16K_MONO:
            return self.derived
        for asset in self.derivatives:
            if asset.role is role:
                return asset
        if role is AudioRole.ASR_16K_MONO and self.master is None:
            return self.derived
        raise KeyError(role.value)

    def lineage_assets(self) -> tuple[AudioAsset, ...]:
        """Return all immutable assets participating in split-leakage checks."""

        if self.master is None:
            return (self.original, self.derived)
        return (self.original, self.master, *self.derivatives)


FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class AcousticSnapshot(StrictSchema):
    """Compact browser-ready view of the derived audio."""

    sample_rate_hz: Literal[16000] = 16_000
    analyzed_duration_ms: FiniteFloat = Field(gt=0)
    truncated: bool
    waveform_times_ms: tuple[FiniteFloat, ...]
    waveform_min: tuple[FiniteFloat, ...]
    waveform_max: tuple[FiniteFloat, ...]
    mel_times_ms: tuple[FiniteFloat, ...]
    mel_frequencies_hz: tuple[FiniteFloat, ...]
    mel_db: tuple[tuple[FiniteFloat, ...], ...]
    f0_times_ms: tuple[FiniteFloat, ...]
    f0_hz: tuple[FiniteFloat | None, ...]
    f0_confidence: tuple[FiniteFloat, ...]
