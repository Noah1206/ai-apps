"""Standalone validation for Audio Lab benchmark and prediction contracts."""

from __future__ import annotations

import hashlib
import io
import json
import stat
import wave
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

EXPECTED_SAMPLE_RATE = 16_000
EXPECTED_CHANNELS = 1
EXPECTED_SAMPLE_WIDTH = 2
BENCHMARK_MANIFEST_NAME = "benchmark.json"
MAX_MEMBER_BYTES = 512 * 1024 * 1024
SHA256_LENGTH = 64


class ContractError(ValueError):
    """Raised when an external package violates a frozen contract."""


@dataclass(frozen=True, slots=True)
class BenchmarkAudio:
    utterance_id: str
    audio_sha256: str
    archive_path: str
    duration_ms: float
    sample_rate: int
    channels: int


@dataclass(frozen=True, slots=True)
class ValidatedBenchmark:
    benchmark_id: str
    benchmark_version: str
    package_sha256: str
    entries: tuple[BenchmarkAudio, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_benchmark_package(
    package_path: Path,
    *,
    expected_benchmark_id: str,
    expected_benchmark_version: str,
    expected_utterances: int,
    schema_path: Path | None = None,
) -> ValidatedBenchmark:
    package_path = package_path.expanduser().resolve()
    if not package_path.is_file():
        raise ContractError(f"benchmark package is missing: {package_path}")
    if not zipfile.is_zipfile(package_path):
        raise ContractError("benchmark package is not a ZIP archive")

    package_hash = sha256_file(package_path)
    try:
        with zipfile.ZipFile(package_path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ContractError("benchmark ZIP contains duplicate member names")
            for info in infos:
                _validate_archive_member(info)
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ContractError(f"benchmark ZIP member failed CRC: {bad_member}")
            if names.count(BENCHMARK_MANIFEST_NAME) != 1:
                raise ContractError("benchmark ZIP must contain exactly one benchmark.json")
            try:
                manifest = json.loads(archive.read(BENCHMARK_MANIFEST_NAME))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ContractError("benchmark.json is not valid UTF-8 JSON") from error
            if schema_path is not None:
                _validate_one_json_schema(
                    manifest,
                    schema_path,
                    location="benchmark.json",
                )
            entries = _validate_manifest(
                manifest,
                expected_benchmark_id=expected_benchmark_id,
                expected_benchmark_version=expected_benchmark_version,
                expected_utterances=expected_utterances,
            )
            expected_members = {BENCHMARK_MANIFEST_NAME}
            validated_entries: list[BenchmarkAudio] = []
            for entry in entries:
                audio_path = _require_string(entry, "derived_audio_path")
                expected_members.add(audio_path)
                try:
                    audio_bytes = archive.read(audio_path)
                except KeyError as error:
                    raise ContractError(f"benchmark audio is missing: {audio_path}") from error
                expected_hash = _require_sha256(entry, "derived_audio_sha256")
                actual_hash = hashlib.sha256(audio_bytes).hexdigest()
                if actual_hash != expected_hash:
                    raise ContractError(f"benchmark audio hash mismatch: {audio_path}")
                audio = _inspect_wav(audio_bytes, audio_path)
                validated_entries.append(
                    BenchmarkAudio(
                        utterance_id=_require_string(entry, "utterance_id"),
                        audio_sha256=expected_hash,
                        archive_path=audio_path,
                        duration_ms=audio["duration_ms"],
                        sample_rate=audio["sample_rate"],
                        channels=audio["channels"],
                    )
                )
            unexpected = set(names) - expected_members
            missing = expected_members - set(names)
            if missing:
                raise ContractError(f"benchmark ZIP is missing manifest audio: {sorted(missing)}")
            if unexpected:
                raise ContractError(
                    f"benchmark ZIP contains unmanifested files: {sorted(unexpected)}"
                )
    except zipfile.BadZipFile as error:
        raise ContractError("benchmark ZIP is corrupt") from error

    return ValidatedBenchmark(
        benchmark_id=expected_benchmark_id,
        benchmark_version=expected_benchmark_version,
        package_sha256=package_hash,
        entries=tuple(validated_entries),
    )


def extract_validated_audio(
    package_path: Path,
    benchmark: ValidatedBenchmark,
    output_dir: Path,
) -> dict[str, Path]:
    """Extract only already-validated audio members under one controlled directory."""

    root = output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    extracted: dict[str, Path] = {}
    with zipfile.ZipFile(package_path.expanduser().resolve()) as archive:
        for entry in benchmark.entries:
            destination = (root / Path(*PurePosixPath(entry.archive_path).parts)).resolve()
            if not destination.is_relative_to(root):
                raise ContractError(f"audio path escapes extraction root: {entry.archive_path}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            payload = archive.read(entry.archive_path)
            if hashlib.sha256(payload).hexdigest() != entry.audio_sha256:
                raise ContractError(f"audio changed during extraction: {entry.archive_path}")
            destination.write_bytes(payload)
            extracted[entry.utterance_id] = destination
    return extracted


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ContractError(f"prediction JSONL contains a blank line at {line_number}")
        try:
            document = json.loads(line)
        except json.JSONDecodeError as error:
            raise ContractError(f"invalid prediction JSON at line {line_number}") from error
        if not isinstance(document, dict):
            raise ContractError(f"prediction line {line_number} is not an object")
        documents.append(document)
    if not documents:
        raise ContractError("prediction JSONL is empty")
    return documents


def validate_prediction_documents(
    documents: Iterable[Mapping[str, Any]],
    *,
    schema_path: Path | None = None,
    benchmark: ValidatedBenchmark | None = None,
) -> list[Mapping[str, Any]]:
    records = list(documents)
    if schema_path is not None:
        _validate_with_json_schema(records, schema_path)

    utterance_ids: list[str] = []
    audio_hashes: list[str] = []
    experiment_ids: set[str] = set()
    models: set[str] = set()
    for index, record in enumerate(records, start=1):
        utterance_id = _require_string(record, "utterance_id", location=f"prediction {index}")
        audio_hash = _require_sha256(record, "audio_sha256", location=f"prediction {index}")
        experiment_ids.add(_require_string(record, "experiment_id", location=f"prediction {index}"))
        result = _require_mapping(record, "result", location=f"prediction {index}")
        surface_text = result.get("surface_text")
        if not isinstance(surface_text, str):
            raise ContractError(f"prediction {index}.result.surface_text must be a string")
        confidence = result.get("confidence")
        confidence_supported = result.get("confidence_supported")
        if confidence_supported is False and confidence is not None:
            raise ContractError(
                f"prediction {index} has confidence while confidence_supported is false"
            )
        if confidence_supported is True and not isinstance(confidence, (int, float)):
            raise ContractError(
                f"prediction {index} must contain confidence when confidence_supported is true"
            )
        segments = result.get("segments")
        if not isinstance(segments, list):
            raise ContractError(f"prediction {index}.result.segments must be an array")
        model = _require_mapping(result, "model", location=f"prediction {index}.result")
        models.add(json.dumps(model, sort_keys=True, ensure_ascii=False))
        utterance_ids.append(utterance_id)
        audio_hashes.append(audio_hash)

    duplicates = _duplicates(utterance_ids)
    if duplicates:
        raise ContractError(f"prediction JSONL has duplicate utterance IDs: {duplicates}")
    duplicate_hashes = _duplicates(audio_hashes)
    if duplicate_hashes:
        raise ContractError(f"prediction JSONL has duplicate audio hashes: {duplicate_hashes}")
    if len(experiment_ids) != 1:
        raise ContractError("prediction JSONL must contain exactly one experiment ID")
    if len(models) != 1:
        raise ContractError("prediction JSONL must contain exactly one model descriptor")

    if benchmark is not None:
        expected = {(entry.utterance_id, entry.audio_sha256) for entry in benchmark.entries}
        actual = set(zip(utterance_ids, audio_hashes, strict=True))
        if actual != expected:
            raise ContractError("prediction JSONL must exactly match benchmark entries")
        for index, record in enumerate(records, start=1):
            if record.get("benchmark_id") != benchmark.benchmark_id:
                raise ContractError(f"prediction {index} benchmark_id does not match")
            if record.get("benchmark_version") != benchmark.benchmark_version:
                raise ContractError(f"prediction {index} benchmark_version does not match")
    return records


def validate_nemotron_prediction_metadata(
    records: Iterable[Mapping[str, Any]],
    *,
    model_id: str,
    resolved_revision: str,
    target_language: str,
) -> None:
    for index, record in enumerate(records, start=1):
        result = _require_mapping(record, "result", location=f"prediction {index}")
        model = _require_mapping(result, "model", location=f"prediction {index}.result")
        expected = {
            "name": model_id,
            "version": resolved_revision,
            "model_provider": "NVIDIA",
            "model_family": "FastConformer-RNNT",
            "decoder_type": "RNNT",
            "target_language": target_language,
            "fine_tuned": False,
        }
        for key, expected_value in expected.items():
            if model.get(key) != expected_value:
                raise ContractError(
                    f"prediction {index}.result.model.{key} must be {expected_value!r}"
                )
        if result.get("confidence") is not None or result.get("confidence_supported") is not False:
            raise ContractError(
                f"prediction {index} must record unsupported confidence as null/false"
            )


def _validate_manifest(
    manifest: Any,
    *,
    expected_benchmark_id: str,
    expected_benchmark_version: str,
    expected_utterances: int,
) -> list[Mapping[str, Any]]:
    if not isinstance(manifest, dict):
        raise ContractError("benchmark.json must contain one object")
    if manifest.get("benchmark_id") != expected_benchmark_id:
        raise ContractError("unexpected benchmark ID")
    if manifest.get("benchmark_version") != expected_benchmark_version:
        raise ContractError("unexpected benchmark version")
    if manifest.get("frozen") is not True:
        raise ContractError("benchmark must be frozen")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ContractError("benchmark entries must be an array")
    if len(entries) != expected_utterances:
        raise ContractError(
            f"expected {expected_utterances} benchmark entries, found {len(entries)}"
        )
    if not all(isinstance(entry, dict) for entry in entries):
        raise ContractError("every benchmark entry must be an object")
    utterance_ids = [_require_string(entry, "utterance_id") for entry in entries]
    duplicates = _duplicates(utterance_ids)
    if duplicates:
        raise ContractError(f"duplicate utterance IDs: {duplicates}")
    paths = [_require_string(entry, "derived_audio_path") for entry in entries]
    duplicate_paths = _duplicates(paths)
    if duplicate_paths:
        raise ContractError(f"duplicate derived audio paths: {duplicate_paths}")
    for path in paths:
        _validate_relative_audio_path(path)
    return entries


def _validate_archive_member(info: zipfile.ZipInfo) -> None:
    _validate_relative_path(info.filename)
    if info.file_size > MAX_MEMBER_BYTES:
        raise ContractError(f"benchmark ZIP member is too large: {info.filename}")
    unix_mode = info.external_attr >> 16
    if unix_mode and stat.S_ISLNK(unix_mode):
        raise ContractError(f"benchmark ZIP may not contain symlinks: {info.filename}")


def _validate_relative_audio_path(path: str) -> None:
    _validate_relative_path(path)
    pure = PurePosixPath(path)
    if pure.suffix.lower() != ".wav":
        raise ContractError(f"benchmark audio is not WAV: {path}")
    if pure.parts[:2] != ("derived", "asr_16k_mono"):
        raise ContractError(f"benchmark audio path is outside derived/asr_16k_mono: {path}")


def _validate_relative_path(path: str) -> None:
    if "\\" in path:
        raise ContractError(f"ZIP member uses a backslash path: {path}")
    pure = PurePosixPath(path)
    if not path or pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise ContractError(f"unsafe relative path: {path}")
    if pure.parts and ":" in pure.parts[0]:
        raise ContractError(f"unsafe drive-qualified path: {path}")


def _inspect_wav(payload: bytes, path: str) -> dict[str, int | float]:
    if not payload:
        raise ContractError(f"benchmark audio is empty: {path}")
    try:
        with wave.open(io.BytesIO(payload), "rb") as audio:
            channels = audio.getnchannels()
            sample_rate = audio.getframerate()
            sample_width = audio.getsampwidth()
            frame_count = audio.getnframes()
            compression = audio.getcomptype()
            frames = audio.readframes(frame_count)
    except (EOFError, wave.Error) as error:
        raise ContractError(f"benchmark audio is not a readable WAV: {path}") from error
    if compression != "NONE":
        raise ContractError(f"benchmark audio must be uncompressed PCM WAV: {path}")
    if channels != EXPECTED_CHANNELS:
        raise ContractError(f"benchmark audio must be mono: {path}")
    if sample_rate != EXPECTED_SAMPLE_RATE:
        raise ContractError(f"benchmark audio must be 16000 Hz: {path}")
    if sample_width != EXPECTED_SAMPLE_WIDTH:
        raise ContractError(f"benchmark audio must be PCM16: {path}")
    if frame_count <= 0 or not frames:
        raise ContractError(f"benchmark audio has no frames: {path}")
    return {
        "channels": channels,
        "sample_rate": sample_rate,
        "duration_ms": frame_count / sample_rate * 1000,
    }


def _validate_with_json_schema(
    records: Iterable[Mapping[str, Any]],
    schema_path: Path,
) -> None:
    for index, record in enumerate(records, start=1):
        _validate_one_json_schema(
            record,
            schema_path,
            location=f"prediction {index}",
        )


def _validate_one_json_schema(
    document: Mapping[str, Any],
    schema_path: Path,
    *,
    location: str,
) -> None:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as error:
        raise ContractError(
            "jsonschema is required; install this package's requirements.txt"
        ) from error
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(document), key=lambda item: list(item.path))
    if errors:
        first = errors[0]
        field = ".".join(str(part) for part in first.absolute_path) or "<root>"
        raise ContractError(f"{location} violates JSON Schema at {field}: {first.message}")


def _require_mapping(
    mapping: Mapping[str, Any],
    key: str,
    *,
    location: str = "object",
) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ContractError(f"{location}.{key} must be an object")
    return value


def _require_string(
    mapping: Mapping[str, Any],
    key: str,
    *,
    location: str = "object",
) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{location}.{key} must be a non-empty string")
    return value


def _require_sha256(
    mapping: Mapping[str, Any],
    key: str,
    *,
    location: str = "object",
) -> str:
    value = _require_string(mapping, key, location=location)
    if len(value) != SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ContractError(f"{location}.{key} must be a lowercase SHA-256")
    return value


def _duplicates(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)
