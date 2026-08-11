"""TASK-004 training-data assembly, leakage validation, and export."""

from __future__ import annotations

import json
import os
import unicodedata
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from busan_lab.audio import hash_file
from busan_lab.schemas.audio import AudioRole
from busan_lab.schemas.benchmark import BenchmarkManifest
from busan_lab.schemas.common import DatasetSplit, ReviewStatus
from busan_lab.schemas.training import (
    DuplicateSurfaceTextGroup,
    TrainingDatasetEntry,
    TrainingDatasetManifest,
    TrainingDatasetValidationReport,
    TrainingExportRecord,
    TrainingSplitAssignments,
)
from busan_lab.schemas.utterance import LabelRevision, UtteranceRecord
from busan_lab.storage import file_is_dataless

if TYPE_CHECKING:
    from busan_lab.storage import LabStorage


class TrainingDatasetValidationError(ValueError):
    """Raised when a candidate dataset would violate the training contract."""

    def __init__(self, report: TrainingDatasetValidationReport) -> None:
        self.report = report
        super().__init__("; ".join(report.errors))


def build_training_dataset(
    *,
    dataset_id: str,
    dataset_version: str,
    records: Iterable[UtteranceRecord],
    assignments: TrainingSplitAssignments,
    benchmark_manifests: Iterable[BenchmarkManifest],
) -> tuple[TrainingDatasetManifest, TrainingDatasetValidationReport]:
    """Build one immutable train/validation manifest from eligible records."""

    records_by_id = {record.utterance_id: record for record in records}
    requested_ids = {
        *assignments.train_utterance_ids,
        *assignments.validation_utterance_ids,
    }
    missing = sorted(str(value) for value in requested_ids - records_by_id.keys())
    if missing:
        raise ValueError(f"training split assignments reference missing utterances: {missing}")

    split_by_id = {
        **{
            utterance_id: DatasetSplit.TRAIN
            for utterance_id in assignments.train_utterance_ids
        },
        **{
            utterance_id: DatasetSplit.VALIDATION
            for utterance_id in assignments.validation_utterance_ids
        },
    }
    entries = tuple(
        _training_entry(records_by_id[utterance_id], split_by_id[utterance_id])
        for utterance_id in (
            *assignments.train_utterance_ids,
            *assignments.validation_utterance_ids,
        )
    )
    manifest = TrainingDatasetManifest(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        entries=entries,
    )
    report = validate_training_dataset(manifest, benchmark_manifests)
    if not report.passed:
        raise TrainingDatasetValidationError(report)
    return manifest, report


def validate_training_dataset(
    manifest: TrainingDatasetManifest,
    benchmark_manifests: Iterable[BenchmarkManifest],
) -> TrainingDatasetValidationReport:
    """Compare a training manifest against every frozen benchmark."""

    frozen_benchmarks = tuple(
        benchmark for benchmark in benchmark_manifests if benchmark.frozen
    )
    checked = tuple(
        sorted(
            f"{benchmark.benchmark_id}@{benchmark.benchmark_version}"
            for benchmark in frozen_benchmarks
        )
    )
    errors: list[str] = []
    warnings: list[str] = []
    if "busan-surface-v0@1.0.0" not in checked:
        errors.append("required frozen benchmark busan-surface-v0@1.0.0 was not checked")

    benchmark_utterances: dict[UUID, str] = {}
    benchmark_speakers: dict[str, str] = {}
    benchmark_audio: dict[str, str] = {}
    benchmark_surfaces: dict[str, str] = {}
    for benchmark in frozen_benchmarks:
        benchmark_ref = f"{benchmark.benchmark_id}@{benchmark.benchmark_version}"
        for benchmark_entry in benchmark.entries:
            benchmark_utterances[benchmark_entry.utterance_id] = benchmark_ref
            benchmark_speakers[benchmark_entry.speaker_id] = benchmark_ref
            for audio_hash in (
                benchmark_entry.original_audio_sha256,
                benchmark_entry.derived_audio_sha256,
                *benchmark_entry.lineage_audio_sha256s,
            ):
                benchmark_audio[audio_hash] = benchmark_ref
            benchmark_surfaces[_surface_key(benchmark_entry.surface_text)] = benchmark_ref

    for training_entry in manifest.entries:
        if utterance_benchmark_ref := benchmark_utterances.get(
            training_entry.utterance_id
        ):
            errors.append(
                f"utterance {training_entry.utterance_id} leaks from frozen benchmark "
                f"{utterance_benchmark_ref}"
            )
        if speaker_benchmark_ref := benchmark_speakers.get(training_entry.speaker_id):
            errors.append(
                f"speaker {training_entry.speaker_id!r} leaks from frozen benchmark "
                f"{speaker_benchmark_ref}"
            )
        overlapping_hashes = sorted(
            set(training_entry.audio_lineage_sha256s).intersection(benchmark_audio)
        )
        if overlapping_hashes:
            benchmark_refs = sorted(
                {benchmark_audio[audio_hash] for audio_hash in overlapping_hashes}
            )
            errors.append(
                f"audio lineage for {training_entry.utterance_id} leaks from frozen "
                "benchmark "
                f"{benchmark_refs}: {overlapping_hashes}"
            )
        if surface_benchmark_ref := benchmark_surfaces.get(
            _surface_key(training_entry.surface_text)
        ):
            errors.append(
                f"exact Surface transcript for {training_entry.utterance_id} leaks from "
                f"frozen benchmark {surface_benchmark_ref}"
            )

    duplicate_groups = _duplicate_surface_groups(manifest.entries)
    if duplicate_groups:
        warnings.append(
            "training data contains repeated Surface transcripts; keep them only when "
            "they represent intentional multi-speaker or multi-environment coverage"
        )
    for group in duplicate_groups:
        if set(group.splits) == {DatasetSplit.TRAIN, DatasetSplit.VALIDATION}:
            errors.append(
                "exact Surface transcript crosses train/validation: "
                f"{group.normalized_surface_text!r}"
            )
    if len(manifest.entries) < 300:
        warnings.append(
            f"Pilot collection target is 300-500 utterances; current count is "
            f"{len(manifest.entries)}"
        )
    train_entries = tuple(
        entry for entry in manifest.entries if entry.split is DatasetSplit.TRAIN
    )
    validation_entries = tuple(
        entry for entry in manifest.entries if entry.split is DatasetSplit.VALIDATION
    )
    if len({entry.speaker_id for entry in validation_entries}) < 2:
        warnings.append("validation should contain at least two independent speakers")

    return TrainingDatasetValidationReport(
        dataset_id=manifest.dataset_id,
        dataset_version=manifest.dataset_version,
        passed=not errors,
        benchmark_manifests_checked=checked,
        train_utterance_count=len(train_entries),
        validation_utterance_count=len(validation_entries),
        train_speaker_count=len({entry.speaker_id for entry in train_entries}),
        validation_speaker_count=len(
            {entry.speaker_id for entry in validation_entries}
        ),
        total_duration_hours=round(
            sum(entry.duration_ms for entry in manifest.entries) / 3_600_000,
            6,
        ),
        duplicate_surface_text_groups=duplicate_groups,
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(warnings),
    )


def review_training_label(
    storage: LabStorage,
    *,
    utterance_id: UUID,
    reviewer_id: str,
    status: ReviewStatus,
    reason: str | None,
) -> UtteranceRecord:
    """Append an eligible training-label review or re-recording revision."""

    if status not in {
        ReviewStatus.HUMAN_REVIEWED,
        ReviewStatus.APPROVED,
        ReviewStatus.DEPRECATED,
    }:
        raise ValueError(
            "training label status must be human_reviewed, approved, or deprecated"
        )
    record = storage.load_utterance(utterance_id)
    if storage.utterance_is_frozen(utterance_id):
        raise ValueError(
            "labels in a frozen benchmark or training dataset cannot be reviewed"
        )
    if record.ground_truth.label_status is status:
        raise ValueError(f"label is already {status.value}")
    revision_number = len(storage.list_label_revisions(utterance_id)) + 1
    updated_ground_truth = record.ground_truth.model_copy(
        update={
            "label_status": status,
            "label_version": f"label_v{revision_number}",
            "reviewer_id": reviewer_id,
        }
    )
    revision = LabelRevision(
        utterance_id=utterance_id,
        previous=record.ground_truth,
        updated=updated_ground_truth,
        changed_by=reviewer_id,
        reason=reason,
    )
    updated_record = record.model_copy(update={"ground_truth": updated_ground_truth})
    storage.save_label_revision(revision)
    storage.save_utterance(updated_record)
    return updated_record


def export_training_dataset_bundle(
    storage: LabStorage,
    manifest: TrainingDatasetManifest,
    report: TrainingDatasetValidationReport,
    output_path: Path,
) -> Path:
    """Write a model-neutral, hash-verified training ZIP without overwriting."""

    if not report.passed:
        raise ValueError("training dataset cannot be exported with validation errors")
    if (
        report.dataset_id != manifest.dataset_id
        or report.dataset_version != manifest.dataset_version
    ):
        raise ValueError("training validation report does not match the manifest")
    destination = output_path.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite training export: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_STORED,
        ) as archive:
            archive.writestr(
                "training_dataset.json",
                manifest.model_dump_json(indent=2),
            )
            archive.writestr(
                "validation_report.json",
                report.model_dump_json(indent=2),
            )
            archive.writestr(
                "schemas/training-dataset.schema.json",
                json.dumps(
                    TrainingDatasetManifest.model_json_schema(),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
            )
            archive.writestr(
                "schemas/training-export-record.schema.json",
                json.dumps(
                    TrainingExportRecord.model_json_schema(),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
            )
            for split in (DatasetSplit.TRAIN, DatasetSplit.VALIDATION):
                rows = tuple(
                    _export_record(entry)
                    for entry in manifest.entries
                    if entry.split is split
                )
                archive.writestr(
                    f"manifests/{split.value}.jsonl",
                    "".join(f"{row.model_dump_json()}\n" for row in rows),
                )
            for entry in manifest.entries:
                audio_path = storage.resolve(entry.asr_audio_path)
                if not audio_path.is_file():
                    raise FileNotFoundError(f"training ASR audio is missing: {audio_path}")
                if file_is_dataless(audio_path):
                    raise OSError(
                        f"training ASR audio is cloud-only: {audio_path}. "
                        "Download data/lab to this Mac and retry."
                    )
                if hash_file(audio_path) != entry.asr_audio_sha256:
                    raise ValueError(f"training ASR audio hash mismatch: {audio_path}")
                archive.write(audio_path, entry.asr_audio_path)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _training_entry(
    record: UtteranceRecord,
    split: DatasetSplit,
) -> TrainingDatasetEntry:
    if not record.consent.model_training_allowed:
        raise ValueError(
            f"utterance {record.utterance_id} has no explicit model-training consent"
        )
    if record.ground_truth.label_status not in {
        ReviewStatus.HUMAN_REVIEWED,
        ReviewStatus.APPROVED,
    }:
        raise ValueError(
            f"utterance {record.utterance_id} label is "
            f"{record.ground_truth.label_status.value}; human review is required"
        )
    if not record.audio.quality.passed:
        raise ValueError(f"utterance {record.utterance_id} failed audio quality checks")
    asr_audio = record.audio.asset_for_role(AudioRole.ASR_16K_MONO)
    lineage_hashes = tuple(
        dict.fromkeys(asset.sha256 for asset in record.audio.lineage_assets())
    )
    return TrainingDatasetEntry(
        utterance_id=record.utterance_id,
        speaker_id=record.speaker.speaker_id,
        split=split,
        region=record.speaker.region,
        recording_environment=record.speaker.environment,
        original_audio_sha256=record.audio.original.sha256,
        asr_audio_sha256=asr_audio.sha256,
        asr_audio_path=asr_audio.relative_path,
        audio_lineage_sha256s=lineage_hashes,
        duration_ms=asr_audio.duration_ms,
        surface_text=record.ground_truth.surface_text,
        label_status=record.ground_truth.label_status,
        label_version=record.ground_truth.label_version,
        dialect_expressions=record.ground_truth.dialect_expressions,
        consent_policy_version=record.consent.policy_version,
    )


def _export_record(entry: TrainingDatasetEntry) -> TrainingExportRecord:
    return TrainingExportRecord(
        utterance_id=entry.utterance_id,
        split=entry.split,
        audio_filepath=entry.asr_audio_path,
        audio_sha256=entry.asr_audio_sha256,
        duration_seconds=round(entry.duration_ms / 1000, 6),
        text=entry.surface_text,
        speaker_id=entry.speaker_id,
        dialect_expressions=tuple(
            expression.surface_form for expression in entry.dialect_expressions
        ),
    )


def _duplicate_surface_groups(
    entries: tuple[TrainingDatasetEntry, ...],
) -> tuple[DuplicateSurfaceTextGroup, ...]:
    grouped: dict[str, list[TrainingDatasetEntry]] = defaultdict(list)
    for entry in entries:
        grouped[_surface_key(entry.surface_text)].append(entry)
    return tuple(
        DuplicateSurfaceTextGroup(
            normalized_surface_text=key,
            utterance_ids=tuple(entry.utterance_id for entry in grouped_entries),
            splits=tuple(
                sorted(
                    {entry.split for entry in grouped_entries},
                    key=lambda split: split.value,
                )
            ),
        )
        for key, grouped_entries in sorted(grouped.items())
        if len(grouped_entries) > 1
    )


def _surface_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())
