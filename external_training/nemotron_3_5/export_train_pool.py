#!/usr/bin/env python3
"""Export one approved TASK-004 import as a train-only GPU handoff ZIP."""

from __future__ import annotations

import argparse
import json
import unicodedata
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from busan_lab.audio import hash_file
from busan_lab.schemas.common import DatasetSplit, ReviewStatus
from busan_lab.storage import LabStorage, file_is_dataless

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
MODEL_REVISION = "f3d333391852ba876df169dcc9ba902d25b6ab0b"
NEMO_REVISION = "6c57e73e83de967eed4d334c493ac313b9afd147"


def _surface_key(value: str) -> str:
    return "".join(unicodedata.normalize("NFKC", value).split()).casefold()


def export_train_pool(
    *,
    data_root: Path,
    import_id: str,
    expected_count: int,
    output: Path,
) -> Path:
    storage = LabStorage(data_root)
    imported = storage.load_training_recording_import(import_id)
    benchmarks = tuple(item for item in storage.list_manifests() if item.frozen)
    benchmark_refs = {
        f"{item.benchmark_id}@{item.benchmark_version}" for item in benchmarks
    }
    if "busan-surface-v0@1.0.0" not in benchmark_refs:
        raise ValueError("required frozen benchmark busan-surface-v0@1.0.0 is missing")
    if len(imported.entries) != expected_count:
        raise ValueError(
            f"expected {expected_count} imported recordings, found {len(imported.entries)}"
        )

    benchmark_ids = {
        entry.utterance_id for benchmark in benchmarks for entry in benchmark.entries
    }
    benchmark_speakers = {
        entry.speaker_id for benchmark in benchmarks for entry in benchmark.entries
    }
    benchmark_hashes = {
        value
        for benchmark in benchmarks
        for entry in benchmark.entries
        for value in (
            entry.original_audio_sha256,
            entry.derived_audio_sha256,
            *entry.lineage_audio_sha256s,
        )
    }
    benchmark_surfaces = {
        _surface_key(entry.surface_text)
        for benchmark in benchmarks
        for entry in benchmark.entries
    }
    if imported.speaker_id in benchmark_speakers:
        raise ValueError("training speaker appears in a frozen benchmark")

    rows: list[dict[str, object]] = []
    audio_sources: list[tuple[Path, str]] = []
    utterance_ids: set[str] = set()
    audio_hashes: set[str] = set()
    total_duration_seconds = 0.0
    for import_entry in imported.entries:
        record = storage.load_utterance(import_entry.utterance_id)
        utterance_id = str(record.utterance_id)
        if record.utterance_id in benchmark_ids:
            raise ValueError(f"benchmark utterance leaked into train pool: {utterance_id}")
        if record.speaker.speaker_id != imported.speaker_id:
            raise ValueError(f"speaker mismatch: {utterance_id}")
        if record.dataset_split is not DatasetSplit.TRAIN:
            raise ValueError(f"record is not assigned to train: {utterance_id}")
        if record.ground_truth.label_status is not ReviewStatus.APPROVED:
            raise ValueError(f"record is not approved: {utterance_id}")
        if not (
            record.consent.storage_allowed
            and record.consent.research_use_allowed
            and record.consent.model_training_allowed
        ):
            raise ValueError(f"training consent is incomplete: {utterance_id}")
        if not record.audio.quality.passed:
            raise ValueError(f"audio quality did not pass: {utterance_id}")
        lineage_hashes = {asset.sha256 for asset in record.audio.lineage_assets()}
        if lineage_hashes.intersection(benchmark_hashes):
            raise ValueError(f"benchmark audio lineage leaked into train pool: {utterance_id}")
        if _surface_key(record.ground_truth.surface_text) in benchmark_surfaces:
            raise ValueError(f"benchmark Surface transcript leaked into train pool: {utterance_id}")

        audio = record.audio.derived
        source = storage.resolve(audio.relative_path)
        if not source.is_file() or file_is_dataless(source):
            raise FileNotFoundError(f"training audio is unavailable: {audio.relative_path}")
        actual_hash = hash_file(source)
        if actual_hash != audio.sha256:
            raise ValueError(f"training audio hash mismatch: {utterance_id}")
        if utterance_id in utterance_ids or actual_hash in audio_hashes:
            raise ValueError(f"duplicate utterance or audio in train pool: {utterance_id}")
        utterance_ids.add(utterance_id)
        audio_hashes.add(actual_hash)

        archive_path = f"audio/{utterance_id}.wav"
        duration_seconds = audio.duration_ms / 1000
        rows.append(
            {
                "audio_filepath": archive_path,
                "duration": duration_seconds,
                "text": record.ground_truth.surface_text,
                "target_lang": "ko-KR",
                "speaker_id": record.speaker.speaker_id,
                "utterance_id": utterance_id,
                "audio_sha256": actual_hash,
                "split": "train",
            }
        )
        audio_sources.append((source, archive_path))
        total_duration_seconds += duration_seconds

    destination = output.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.unlink(missing_ok=True)
    metadata = {
        "schema_version": "1.0.0",
        "task_id": "TASK-005",
        "package_type": "approved_train_pool",
        "status": "train_pool_ready_validation_missing",
        "training_permitted": False,
        "validation_required": True,
        "created_at": datetime.now(UTC).isoformat(),
        "source_import_id": import_id,
        "speaker_id": imported.speaker_id,
        "train_utterance_count": len(rows),
        "validation_utterance_count": 0,
        "total_duration_seconds": round(total_duration_seconds, 6),
        "target_language": "ko-KR",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "nemo_revision": NEMO_REVISION,
        "benchmarks_excluded": sorted(benchmark_refs),
    }
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr(
                "package_metadata.json",
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            )
            archive.writestr(
                "train_manifest.jsonl",
                "".join(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                    for row in rows
                ),
            )
            archive.writestr(
                "VALIDATION_REQUIRED.txt",
                "This package contains Train data only. Do not start fine-tuning until "
                "an independent-speaker Validation package is available.\n",
            )
            for source, archive_path in audio_sources:
                archive.write(source, archive_path)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--import-id", required=True)
    parser.add_argument("--expected-count", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = export_train_pool(
        data_root=args.data_root,
        import_id=args.import_id,
        expected_count=args.expected_count,
        output=args.output,
    )
    print(path)


if __name__ == "__main__":
    main()
