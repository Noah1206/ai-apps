"""Build benchmark manifests from immutable utterance records."""

from __future__ import annotations

from collections.abc import Iterable

from busan_lab.schemas.benchmark import BenchmarkEntry, BenchmarkManifest
from busan_lab.schemas.common import DatasetSplit
from busan_lab.schemas.utterance import UtteranceRecord


def build_manifest(
    *,
    benchmark_id: str,
    benchmark_version: str,
    records: Iterable[UtteranceRecord],
    split: DatasetSplit,
) -> BenchmarkManifest:
    if split is DatasetSplit.UNASSIGNED:
        raise ValueError("a benchmark manifest requires an explicit dataset split")
    entries = tuple(
        BenchmarkEntry(
            utterance_id=record.utterance_id,
            speaker_id=record.speaker.speaker_id,
            split=split,
            original_audio_sha256=record.audio.original.sha256,
            derived_audio_sha256=record.audio.derived.sha256,
            derived_audio_path=record.audio.derived.relative_path,
            lineage_audio_sha256s=tuple(asset.sha256 for asset in record.audio.lineage_assets()),
            surface_text=record.ground_truth.surface_text,
            normalized_meaning=record.ground_truth.normalized_meaning,
            dialect_expressions=record.ground_truth.dialect_expressions,
        )
        for record in records
    )
    return BenchmarkManifest(
        benchmark_id=benchmark_id,
        benchmark_version=benchmark_version,
        entries=entries,
    )
