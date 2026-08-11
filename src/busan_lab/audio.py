"""Audio ingestion, validation, immutable preservation, and derivation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from busan_lab.config import LabSettings
from busan_lab.schemas.audio import (
    AudioAsset,
    AudioBundle,
    AudioQualityReport,
    AudioRole,
    AudioTransformation,
)
from busan_lab.storage import LabStorage


class AudioValidationError(ValueError):
    """Raised when uploaded bytes are not an acceptable audio asset."""


@dataclass(frozen=True, slots=True)
class ProbeResult:
    container: str
    codec: str
    sample_rate_hz: int
    channels: int
    sample_width_bits: int | None
    duration_ms: float


@dataclass(frozen=True, slots=True)
class AudioProfile:
    role: AudioRole
    filename_label: str
    sample_rate_hz: int
    channels: int
    codec: str


class AudioProcessor:
    """Preserve uploaded bytes and create a deterministic master audio tree."""

    _PURPOSE_PROFILES = (
        AudioProfile(
            role=AudioRole.ASR_16K_MONO,
            filename_label="asr-16k-mono",
            sample_rate_hz=16_000,
            channels=1,
            codec="pcm_s16le",
        ),
        AudioProfile(
            role=AudioRole.PRONUNCIATION_24K_MONO,
            filename_label="pronunciation-24k-mono",
            sample_rate_hz=24_000,
            channels=1,
            codec="pcm_s16le",
        ),
        AudioProfile(
            role=AudioRole.TTS_48K_MONO,
            filename_label="tts-48k-mono",
            sample_rate_hz=48_000,
            channels=1,
            codec="pcm_s24le",
        ),
    )

    def __init__(self, settings: LabSettings, storage: LabStorage) -> None:
        self.settings = settings
        self.storage = storage

    def process(self, staged_path: Path, original_filename: str) -> AudioBundle:
        if not staged_path.is_file():
            raise AudioValidationError("staged upload does not exist")
        byte_size = staged_path.stat().st_size
        if byte_size <= 0:
            raise AudioValidationError("empty uploads are not audio")
        if byte_size > self.settings.max_upload_bytes:
            raise AudioValidationError(
                f"audio exceeds {self.settings.max_upload_bytes} byte upload limit"
            )

        original_probe = self._probe(staged_path)
        self._validate_probe(original_probe)
        original_sha = hash_file(staged_path)
        suffix = _safe_audio_suffix(original_filename)
        raw_path = self.storage.raw_dir / original_sha[:2] / f"{original_sha}{suffix}"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if not raw_path.exists():
            shutil.copy2(staged_path, raw_path)

        ffmpeg_version = self._ffmpeg_version()
        original_asset = AudioAsset(
            asset_id=f"original-{original_sha[:24]}",
            role=AudioRole.ORIGINAL,
            relative_path=self.storage.relative(raw_path),
            sha256=original_sha,
            byte_size=raw_path.stat().st_size,
            container=original_probe.container,
            codec=original_probe.codec,
            sample_rate_hz=original_probe.sample_rate_hz,
            channels=original_probe.channels,
            sample_width_bits=original_probe.sample_width_bits,
            duration_ms=original_probe.duration_ms,
        )

        master_channels = 1 if original_probe.channels == 1 else 2
        master_role = (
            AudioRole.MASTER_48K_MONO if master_channels == 1 else AudioRole.MASTER_48K_STEREO
        )
        master_profile = AudioProfile(
            role=master_role,
            filename_label=f"master-48k-{'mono' if master_channels == 1 else 'stereo'}",
            sample_rate_hz=48_000,
            channels=master_channels,
            codec="pcm_s24le",
        )
        master_path = (
            self.storage.master_dir
            / original_sha[:2]
            / f"{original_sha}.{master_profile.filename_label}.wav"
        )
        master_asset, master_transformation = self._derive_asset(
            source_path=raw_path,
            parent_sha256=original_sha,
            output_path=master_path,
            profile=master_profile,
            ffmpeg_version=ffmpeg_version,
        )

        derivative_assets: list[AudioAsset] = []
        derivative_transformations: list[AudioTransformation] = []
        for profile in self._PURPOSE_PROFILES:
            output_path = (
                self.storage.derived_dir
                / profile.role.value
                / original_sha[:2]
                / f"{original_sha}.{profile.filename_label}.wav"
            )
            asset, transformation = self._derive_asset(
                source_path=master_path,
                parent_sha256=master_asset.sha256,
                output_path=output_path,
                profile=profile,
                ffmpeg_version=ffmpeg_version,
            )
            derivative_assets.append(asset)
            derivative_transformations.append(transformation)

        asr_asset = next(
            asset for asset in derivative_assets if asset.role is AudioRole.ASR_16K_MONO
        )
        asr_transformation = next(
            transformation
            for transformation in derivative_transformations
            if transformation.target_role is AudioRole.ASR_16K_MONO
        )
        samples = read_pcm16_wav(self.storage.resolve(asr_asset.relative_path))
        quality = assess_quality(
            samples,
            asr_asset.sample_rate_hz,
            min_duration_ms=self.settings.min_duration_ms,
            max_duration_ms=self.settings.max_duration_ms,
        )

        return AudioBundle(
            original=original_asset,
            master=master_asset,
            derived=asr_asset,
            derivatives=tuple(derivative_assets),
            transformation=asr_transformation,
            transformations=(
                master_transformation,
                *derivative_transformations,
            ),
            quality=quality,
        )

    def _derive_asset(
        self,
        *,
        source_path: Path,
        parent_sha256: str,
        output_path: Path,
        profile: AudioProfile,
        ffmpeg_version: str,
    ) -> tuple[AudioAsset, AudioTransformation]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_args = (
            "-vn",
            "-map_metadata",
            "-1",
            "-ac",
            str(profile.channels),
            "-ar",
            str(profile.sample_rate_hz),
            "-c:a",
            profile.codec,
        )
        if not output_path.exists():
            temporary = output_path.with_suffix(".wav.tmp")
            command = [
                self.settings.ffmpeg_binary,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source_path),
                *ffmpeg_args,
                "-f",
                "wav",
                str(temporary),
            ]
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=180,
            )
            if completed.returncode != 0:
                temporary.unlink(missing_ok=True)
                raise AudioValidationError(
                    f"FFmpeg could not create {profile.role.value}: "
                    f"{completed.stderr.strip()[:400]}"
                )
            os.replace(temporary, output_path)

        probe = self._probe(output_path)
        if (
            probe.sample_rate_hz != profile.sample_rate_hz
            or probe.channels != profile.channels
            or probe.codec != profile.codec
        ):
            raise AudioValidationError(
                f"{profile.role.value} did not satisfy "
                f"{profile.sample_rate_hz}Hz/{profile.channels}ch/{profile.codec}"
            )
        asset_sha = hash_file(output_path)
        transformation = AudioTransformation(
            name=f"ffmpeg_{profile.role.value}_{profile.codec}",
            target_role=profile.role,
            tool_version=ffmpeg_version,
            command_fingerprint=hashlib.sha256("\0".join(ffmpeg_args).encode()).hexdigest(),
        )
        asset = AudioAsset(
            asset_id=f"{profile.role.value}-{asset_sha[:24]}",
            role=profile.role,
            relative_path=self.storage.relative(output_path),
            sha256=asset_sha,
            byte_size=output_path.stat().st_size,
            container=probe.container,
            codec=probe.codec,
            sample_rate_hz=probe.sample_rate_hz,
            channels=probe.channels,
            sample_width_bits=probe.sample_width_bits,
            duration_ms=probe.duration_ms,
            parent_sha256=parent_sha256,
        )
        return asset, transformation

    def _validate_probe(self, probe: ProbeResult) -> None:
        if probe.channels < 1 or probe.sample_rate_hz < 1:
            raise AudioValidationError("audio stream has invalid channel or sample-rate metadata")
        if probe.duration_ms < self.settings.min_duration_ms:
            raise AudioValidationError(
                f"audio is shorter than {self.settings.min_duration_ms:.0f} ms"
            )
        if probe.duration_ms > self.settings.max_duration_ms:
            raise AudioValidationError(
                f"audio is longer than {self.settings.max_duration_ms / 1000:.0f} seconds"
            )

    def _probe(self, path: Path) -> ProbeResult:
        command = [
            self.settings.ffprobe_binary,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,bits_per_sample,duration",
            "-show_entries",
            "format=format_name,duration",
            "-of",
            "json",
            str(path),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise AudioValidationError(f"FFprobe is unavailable: {error}") from error
        if completed.returncode != 0:
            raise AudioValidationError(
                f"file is not decodable audio: {completed.stderr.strip()[:300]}"
            )
        try:
            payload: dict[str, Any] = json.loads(completed.stdout)
            stream = payload["streams"][0]
            format_info = payload.get("format", {})
            duration = float(stream.get("duration") or format_info["duration"])
            sample_width = int(stream.get("bits_per_sample") or 0) or None
            return ProbeResult(
                container=str(format_info.get("format_name") or path.suffix.lstrip(".")),
                codec=str(stream["codec_name"]),
                sample_rate_hz=int(stream["sample_rate"]),
                channels=int(stream["channels"]),
                sample_width_bits=sample_width,
                duration_ms=duration * 1000,
            )
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise AudioValidationError("audio metadata is incomplete") from error

    def _ffmpeg_version(self) -> str:
        completed = subprocess.run(
            [self.settings.ffmpeg_binary, "-version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        first_line = completed.stdout.splitlines()[0] if completed.stdout else "unknown"
        return first_line[:200]


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_pcm16_wav(path: Path) -> NDArray[np.float32]:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getframerate() != 16_000 or wav.getsampwidth() != 2:
            raise AudioValidationError("analysis requires a 16kHz mono PCM16 WAV")
        raw = wav.readframes(wav.getnframes())
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    return samples / 32768.0


def assess_quality(
    samples: NDArray[np.float32],
    sample_rate_hz: int,
    *,
    min_duration_ms: float,
    max_duration_ms: float,
) -> AudioQualityReport:
    if samples.size == 0:
        raise AudioValidationError("derived audio has no samples")
    absolute = np.abs(samples)
    peak = float(np.max(absolute))
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    rms_dbfs = 20 * math.log10(max(rms, 1e-12))
    clipping_ratio = float(np.mean(absolute >= 0.999))
    silence_ratio = float(np.mean(absolute < 0.01))
    dc_offset = float(np.mean(samples))
    duration_ms = samples.size / sample_rate_hz * 1000

    warnings: list[str] = []
    if clipping_ratio > 0.01:
        warnings.append("CLIPPING_HIGH")
    if rms_dbfs < -50:
        warnings.append("LEVEL_TOO_LOW")
    if silence_ratio > 0.95:
        warnings.append("MOSTLY_SILENCE")
    if abs(dc_offset) > 0.05:
        warnings.append("DC_OFFSET_HIGH")
    if not min_duration_ms <= duration_ms <= max_duration_ms:
        warnings.append("DURATION_OUT_OF_RANGE")

    return AudioQualityReport(
        passed=not warnings,
        duration_ms=duration_ms,
        peak_amplitude=peak,
        rms_dbfs=rms_dbfs,
        clipping_ratio=clipping_ratio,
        silence_ratio=silence_ratio,
        dc_offset=dc_offset,
        warnings=tuple(warnings),
    )


def _safe_audio_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if 1 < len(suffix) <= 10 and suffix[1:].isalnum():
        return suffix
    return ".audio"
