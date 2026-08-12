#!/usr/bin/env python3
"""Build a local-only Gate 2 Busan test set from the AI Hub archive.

The archive reader and audio slicing helpers are supplied by the recovery
tooling in ``PORTER-main``.  This script adds the Gate 2 exclusion, manifest,
and provenance contracts needed for a final held-out evaluation set.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from busan_lab.evaluation.metrics import normalize_for_cer


def _load_porter(porter_dir: Path) -> Any:
    sys.path.insert(0, str(porter_dir))
    try:
        return importlib.import_module("build_task005_independent_validation")
    finally:
        sys.path.pop(0)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _dialect_expression_index(labels_dir: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for label_path in sorted(labels_dir.glob("*.json")):
        document = _read_json(label_path)
        for utterance in document.get("utterance", []):
            utterance_id = str(utterance.get("id", ""))
            expressions: list[str] = []
            for token in utterance.get("eojeolList", []):
                if token.get("isDialect") is True:
                    expression = str(token.get("eojeol", "")).strip()
                    if expression and expression not in expressions:
                        expressions.append(expression)
            if not expressions:
                dialect = str(utterance.get("dialect_form", "")).strip()
                standard = str(utterance.get("standard_form", "")).strip()
                if dialect and dialect != standard:
                    expressions.append(dialect)
            result[utterance_id] = expressions
    return result


def _filter_pools(porter: Any, pools: list[Any], exclusions: dict[str, Any]) -> list[Any]:
    excluded_speakers = set(map(str, exclusions.get("speaker_ids", [])))
    excluded_utterances = set(map(str, exclusions.get("utterance_ids", [])))
    excluded_recordings = set(map(str, exclusions.get("source_recording_ids", [])))
    excluded_surfaces = set(map(str, exclusions.get("normalized_surface_texts", [])))
    filtered: list[Any] = []
    for pool in pools:
        if pool.global_speaker_id in excluded_speakers:
            continue
        if pool.recording_id in excluded_recordings:
            continue
        candidates = [
            candidate
            for candidate in pool.candidates
            if candidate.utterance_id not in excluded_utterances
            and normalize_for_cer(candidate.dialect_form) not in excluded_surfaces
        ]
        filtered.append(replace(pool, candidates=candidates))
    return filtered


def _choose_speakers(pools: list[Any], count: int, utterances: int) -> list[Any]:
    eligible = [
        pool
        for pool in pools
        if pool.region_rank == 3 and len(pool.candidates) >= utterances
    ]
    eligible.sort(
        key=lambda pool: (-len(pool.candidates), pool.recording_id, pool.local_speaker_id)
    )
    selected: list[Any] = []
    recordings: set[str] = set()
    signatures: set[tuple[str, ...]] = set()
    sex_ages: set[tuple[str, str]] = set()

    def add(pool: Any, *, require_new_sex_age: bool) -> None:
        sex_age = (
            str(pool.speaker_metadata.get("sex", "")),
            str(pool.speaker_metadata.get("age", "")),
        )
        if pool.recording_id in recordings or pool.demographic_signature in signatures:
            return
        if require_new_sex_age and sex_age in sex_ages:
            return
        selected.append(pool)
        recordings.add(pool.recording_id)
        signatures.add(pool.demographic_signature)
        sex_ages.add(sex_age)

    for require_new_sex_age in (True, False):
        for pool in eligible:
            if len(selected) >= count:
                break
            add(pool, require_new_sex_age=require_new_sex_age)
        if len(selected) >= count:
            break
    if len(selected) != count:
        raise ValueError(
            f"only {len(selected)} independent Busan speakers have at least "
            f"{utterances} eligible utterances"
        )
    return selected


def _selection_summary(
    selected_speakers: list[Any],
    selected_utterances: list[Any],
    corpus: dict[str, Any],
) -> dict[str, Any]:
    return {
        "corpus": corpus,
        "selected_speakers": [
            {
                "speaker_id": pool.global_speaker_id,
                "recording_id": pool.recording_id,
                "local_speaker_id": pool.local_speaker_id,
                "eligible_utterances": len(pool.candidates),
                "speaker_metadata": pool.speaker_metadata,
            }
            for pool in selected_speakers
        ],
        "utterance_count": len(selected_utterances),
        "speaker_count": len(selected_speakers),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", required=True, type=Path)
    parser.add_argument("--label-tar", required=True, type=Path)
    parser.add_argument("--source-tar", required=True, type=Path)
    parser.add_argument("--exclusions", required=True, type=Path)
    parser.add_argument("--porter-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--speaker-count", type=int, default=5)
    parser.add_argument("--utterances-per-speaker", type=int, default=20)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    labels_dir = args.labels_dir.resolve(strict=True)
    label_tar = args.label_tar.resolve(strict=True)
    source_tar = args.source_tar.resolve(strict=True)
    exclusion_path = args.exclusions.resolve(strict=True)
    porter_dir = args.porter_dir.resolve(strict=True)
    porter = _load_porter(porter_dir)
    exclusions = _read_json(exclusion_path)

    pools, corpus = porter.load_speaker_pools(labels_dir, set(), set())
    pools = _filter_pools(porter, pools, exclusions)
    selected_speakers = _choose_speakers(
        pools,
        count=args.speaker_count,
        utterances=args.utterances_per_speaker,
    )
    selected_utterances = [
        candidate
        for pool in selected_speakers
        for candidate in porter.choose_evenly(
            pool.candidates,
            count=args.utterances_per_speaker,
        )
    ]
    summary = _selection_summary(selected_speakers, selected_utterances, corpus)
    if args.report_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required unless --report-only is used")

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    audio_dir = output_dir / "audio"
    source_dir = output_dir / "source-recordings"
    output_dir.mkdir(parents=True)
    audio_dir.mkdir()
    expression_index = _dialect_expression_index(labels_dir)
    source_inventory = porter.extract_recordings(
        source_tar,
        [pool.recording_id for pool in selected_speakers],
        source_dir,
    )
    speaker_by_key = {
        (pool.recording_id, pool.local_speaker_id): pool for pool in selected_speakers
    }
    entries: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for candidate in selected_utterances:
        pool = speaker_by_key[(candidate.recording_id, candidate.local_speaker_id)]
        source_info = source_inventory["recordings"][candidate.recording_id]
        if candidate.end > float(source_info["wav"]["duration_seconds"]) + 0.02:
            raise ValueError(f"annotation exceeds source audio: {candidate.utterance_id}")
        output_name = f"{candidate.utterance_id}.wav"
        audio_path = audio_dir / output_name
        porter.convert_utterance(Path(source_info["extracted_path"]), candidate, audio_path)
        wav_info = porter.wav_contract(audio_path)
        expressions = expression_index.get(candidate.utterance_id, [])
        if not expressions:
            raise ValueError(f"missing dialect expression label: {candidate.utterance_id}")
        entries.append(
            {
                "utterance_id": candidate.utterance_id,
                "speaker_id": pool.global_speaker_id,
                "source_recording_id": candidate.recording_id,
                "region": "부산",
                "audio_filepath": f"audio/{output_name}",
                "audio_sha256": porter.sha256_file(audio_path),
                "audio_lineage_sha256s": [source_info["sha256"]],
                "duration_seconds": wav_info["duration_seconds"],
                "surface_text": candidate.dialect_form,
                "dialect_expressions": expressions,
                "label_status": "human_reviewed",
            }
        )
        provenance.append(
            {
                "utterance_id": candidate.utterance_id,
                "source_label_field": "dialect_form",
                "standard_form_not_used_as_reference": candidate.standard_form,
                "source_start_seconds": candidate.start,
                "source_end_seconds": candidate.end,
                "speaker_metadata": pool.speaker_metadata,
            }
        )

    now = dt.datetime.now(dt.UTC).isoformat()
    manifest = {
        "schema_version": "1.0.0",
        "dataset_id": "aihub-gyeongsang-busan-final-test",
        "dataset_version": "1.0.0",
        "dataset_kind": "independent_busan_test",
        "created_at": now,
        "frozen": True,
        "target_language": "ko-KR",
        "training_allowed": False,
        "checkpoint_selection_allowed": False,
        "source_name": "AI Hub 한국어 방언 발화 데이터(경상도), Validation split",
        "source_version": "local archive; AI Hub dataSetSn=119",
        "license_or_access_policy": (
            "AI Hub 데이터 이용정책; local evaluation only; redistribution prohibited "
            "without the required prior consultation"
        ),
        "entries": entries,
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "selection-provenance.json", provenance)
    _write_json(output_dir / "source-inventory.json", source_inventory)
    _write_json(
        output_dir / "build-report.json",
        {
            **summary,
            "created_at": now,
            "label_tar": str(label_tar),
            "label_tar_sha256": porter.sha256_file(label_tar),
            "source_tar": str(source_tar),
            "source_tar_bytes": source_tar.stat().st_size,
            "exclusion_registry": str(exclusion_path),
            "exclusion_registry_sha256": porter.sha256_file(exclusion_path),
            "label_status_basis": (
                "AI Hub human speech labels provide dialect_form, standard_form, and "
                "per-eojeol isDialect annotations; no machine-generated labels are used"
            ),
            "data_handling": "local-only; do not commit or redistribute audio or labels",
            "speaker_counts": dict(
                sorted(collections.Counter(entry["speaker_id"] for entry in entries).items())
            ),
        },
    )
    print(f"manifest={output_dir / 'manifest.json'}")
    print(f"utterances={len(entries)}")
    print(f"speakers={len({entry['speaker_id'] for entry in entries})}")
    print(f"duration_seconds={sum(entry['duration_seconds'] for entry in entries):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
