#!/usr/bin/env python3
"""Build the frozen Train/Validation/Test allocation for the final Gate 2 attempt."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import shutil
import sys
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from busan_lab.evaluation.metrics import normalize_for_cer


@dataclass(frozen=True)
class AllocationSpec:
    name: str
    speakers: int
    utterances_per_speaker: int


@dataclass(frozen=True)
class Allocation:
    spec: AllocationSpec
    speakers: tuple[Any, ...]
    utterances: tuple[Any, ...]


SPECS = (
    AllocationSpec("test-v2", speakers=10, utterances_per_speaker=10),
    AllocationSpec("validation", speakers=10, utterances_per_speaker=10),
    AllocationSpec("train", speakers=40, utterances_per_speaker=20),
)


def _load_porter(porter_dir: Path) -> Any:
    sys.path.insert(0, str(porter_dir))
    try:
        return importlib.import_module("build_task005_independent_validation")
    finally:
        sys.path.pop(0)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _expression_index(labels_dir: Path) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for label_path in sorted(labels_dir.glob("*.json")):
        document = _read_json(label_path)
        for utterance in document.get("utterance", []):
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
            result[str(utterance.get("id", ""))] = tuple(expressions)
    return result


def _initial_exclusions(
    registry: dict[str, Any],
    consumed_test: dict[str, Any],
) -> dict[str, set[str]]:
    result = {
        "speakers": set(map(str, registry.get("speaker_ids", []))),
        "utterances": set(map(str, registry.get("utterance_ids", []))),
        "recordings": set(map(str, registry.get("source_recording_ids", []))),
        "audio": set(map(str, registry.get("audio_sha256s", []))),
        "surfaces": set(map(str, registry.get("normalized_surface_texts", []))),
    }
    for entry in consumed_test.get("entries", []):
        result["speakers"].add(str(entry["speaker_id"]))
        result["utterances"].add(str(entry["utterance_id"]))
        result["recordings"].add(str(entry["source_recording_id"]))
        result["audio"].add(str(entry["audio_sha256"]))
        result["audio"].update(map(str, entry.get("audio_lineage_sha256s", [])))
        result["surfaces"].add(normalize_for_cer(str(entry["surface_text"])))
    return result


def _filter_pools(
    pools: list[Any],
    exclusions: dict[str, set[str]],
    expressions: dict[str, tuple[str, ...]],
) -> list[Any]:
    result: list[Any] = []
    for pool in pools:
        if pool.region_rank != 3:
            continue
        if pool.global_speaker_id in exclusions["speakers"]:
            continue
        if pool.recording_id in exclusions["recordings"]:
            continue
        candidates = [
            candidate
            for candidate in pool.candidates
            if candidate.utterance_id not in exclusions["utterances"]
            and normalize_for_cer(candidate.dialect_form) not in exclusions["surfaces"]
            and expressions.get(candidate.utterance_id)
        ]
        result.append(replace(pool, candidates=candidates))
    result.sort(
        key=lambda pool: (-len(pool.candidates), pool.recording_id, pool.local_speaker_id)
    )
    return result


def _choose_diverse(
    candidates: list[Any],
    expressions: dict[str, tuple[str, ...]],
    count: int,
) -> list[Any]:
    remaining = sorted(candidates, key=lambda item: (item.start, item.utterance_id))
    chosen: list[Any] = []
    seen: set[str] = set()
    while len(chosen) < count:
        if not remaining:
            raise ValueError("not enough unique candidates")

        def score(candidate: Any) -> tuple[int, int, float]:
            labels = {
                normalize_for_cer(value)
                for value in expressions[candidate.utterance_id]
                if normalize_for_cer(value)
            }
            return (len(labels - seen), len(labels), candidate.annotated_duration)

        selected = max(remaining, key=score)
        remaining.remove(selected)
        chosen.append(selected)
        seen.update(
            normalize_for_cer(value)
            for value in expressions[selected.utterance_id]
            if normalize_for_cer(value)
        )
    return sorted(chosen, key=lambda item: (item.start, item.utterance_id))


def _allocate(
    pools: list[Any],
    exclusions: dict[str, set[str]],
    expressions: dict[str, tuple[str, ...]],
) -> tuple[Allocation, ...]:
    used_speakers = set(exclusions["speakers"])
    used_recordings = set(exclusions["recordings"])
    used_utterances = set(exclusions["utterances"])
    used_surfaces = set(exclusions["surfaces"])
    allocations: list[Allocation] = []
    for spec in SPECS:
        selected_pools: list[Any] = []
        selected_utterances: list[Any] = []
        for pool in pools:
            if len(selected_pools) >= spec.speakers:
                break
            if pool.global_speaker_id in used_speakers or pool.recording_id in used_recordings:
                continue
            candidates = [
                candidate
                for candidate in pool.candidates
                if candidate.utterance_id not in used_utterances
                and normalize_for_cer(candidate.dialect_form) not in used_surfaces
            ]
            if len(candidates) < spec.utterances_per_speaker:
                continue
            chosen = _choose_diverse(
                candidates,
                expressions,
                spec.utterances_per_speaker,
            )
            selected_pools.append(pool)
            selected_utterances.extend(chosen)
            used_speakers.add(pool.global_speaker_id)
            used_recordings.add(pool.recording_id)
            used_utterances.update(item.utterance_id for item in chosen)
            used_surfaces.update(normalize_for_cer(item.dialect_form) for item in chosen)
        if len(selected_pools) != spec.speakers:
            raise ValueError(
                f"{spec.name}: only {len(selected_pools)} of {spec.speakers} speakers allocated"
            )
        allocations.append(
            Allocation(spec, tuple(selected_pools), tuple(selected_utterances))
        )
    return tuple(allocations)


def _allocation_report(
    allocations: tuple[Allocation, ...],
    expressions: dict[str, tuple[str, ...]],
    corpus: dict[str, Any],
) -> dict[str, Any]:
    return {
        "corpus": corpus,
        "allocation_order": [spec.name for spec in SPECS],
        "splits": {
            allocation.spec.name: {
                "speakers": len(allocation.speakers),
                "utterances": len(allocation.utterances),
                "dialect_expression_labels": sum(
                    len(expressions[item.utterance_id]) for item in allocation.utterances
                ),
                "selected_speakers": [
                    {
                        "speaker_id": pool.global_speaker_id,
                        "recording_id": pool.recording_id,
                        "eligible_utterances": len(pool.candidates),
                        "metadata": pool.speaker_metadata,
                    }
                    for pool in allocation.speakers
                ],
            }
            for allocation in allocations
        },
    }


def _safe_zip_manifest(
    package: Path,
    manifest_name: str,
    destination_root: Path,
    split: str,
    porter: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(package) as archive:
        names = set(archive.namelist())
        source_rows = [
            json.loads(line)
            for line in archive.read(manifest_name).decode("utf-8").splitlines()
            if line.strip()
        ]
        for source_row in source_rows:
            member_name = str(source_row["audio_filepath"])
            member = PurePosixPath(member_name)
            if member.is_absolute() or ".." in member.parts or "\\" in member_name:
                raise ValueError(f"unsafe package audio path: {member_name}")
            if member_name not in names:
                raise ValueError(f"missing package audio: {member_name}")
            destination = destination_root / "audio" / member.name
            if destination.exists():
                raise FileExistsError(destination)
            with archive.open(member_name) as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
            expected = str(source_row["audio_sha256"])
            if porter.sha256_file(destination) != expected:
                raise ValueError(f"historical package audio hash mismatch: {member_name}")
            porter.wav_contract(destination)
            row = dict(source_row)
            row["audio_filepath"] = f"audio/{member.name}"
            row["split"] = split
            rows.append(row)
    return rows


def _absolute_rows(root: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        converted = dict(row)
        converted["audio_filepath"] = str((root / row["audio_filepath"]).resolve())
        result.append(converted)
    return result


def _new_development_rows(
    allocation: Allocation,
    destination_root: Path,
    source_inventory: dict[str, Any],
    pool_by_key: dict[tuple[str, str], Any],
    expressions: dict[str, tuple[str, ...]],
    porter: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for candidate in allocation.utterances:
        pool = pool_by_key[(candidate.recording_id, candidate.local_speaker_id)]
        output_name = f"{candidate.utterance_id}.wav"
        destination = destination_root / "audio" / output_name
        source_info = source_inventory["recordings"][candidate.recording_id]
        porter.convert_utterance(Path(source_info["extracted_path"]), candidate, destination)
        wav_info = porter.wav_contract(destination)
        rows.append(
            {
                "utterance_id": candidate.utterance_id,
                "audio_filepath": f"audio/{output_name}",
                "duration": wav_info["duration_seconds"],
                "text": candidate.dialect_form,
                "speaker_id": pool.global_speaker_id,
                "target_lang": "ko-KR",
                "split": allocation.spec.name,
                "audio_sha256": porter.sha256_file(destination),
            }
        )
        provenance.append(
            {
                "utterance_id": candidate.utterance_id,
                "speaker_id": pool.global_speaker_id,
                "source_recording_id": candidate.recording_id,
                "source_audio_sha256": source_info["sha256"],
                "source_start_seconds": candidate.start,
                "source_end_seconds": candidate.end,
                "dialect_form": candidate.dialect_form,
                "standard_form_not_used_as_reference": candidate.standard_form,
                "dialect_expressions": list(expressions[candidate.utterance_id]),
                "speaker_metadata": pool.speaker_metadata,
            }
        )
    return rows, provenance


def _test_manifest(
    allocation: Allocation,
    destination_root: Path,
    source_inventory: dict[str, Any],
    pool_by_key: dict[tuple[str, str], Any],
    expressions: dict[str, tuple[str, ...]],
    porter: Any,
    created_at: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    entries: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for candidate in allocation.utterances:
        pool = pool_by_key[(candidate.recording_id, candidate.local_speaker_id)]
        output_name = f"{candidate.utterance_id}.wav"
        destination = destination_root / "audio" / output_name
        source_info = source_inventory["recordings"][candidate.recording_id]
        porter.convert_utterance(Path(source_info["extracted_path"]), candidate, destination)
        wav_info = porter.wav_contract(destination)
        entries.append(
            {
                "utterance_id": candidate.utterance_id,
                "speaker_id": pool.global_speaker_id,
                "source_recording_id": candidate.recording_id,
                "region": "부산",
                "audio_filepath": f"audio/{output_name}",
                "audio_sha256": porter.sha256_file(destination),
                "audio_lineage_sha256s": [source_info["sha256"]],
                "duration_seconds": wav_info["duration_seconds"],
                "surface_text": candidate.dialect_form,
                "dialect_expressions": list(expressions[candidate.utterance_id]),
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
    manifest = {
        "schema_version": "1.0.0",
        "dataset_id": "aihub-gyeongsang-busan-final-test-v2",
        "dataset_version": "2.0.0",
        "dataset_kind": "independent_busan_test",
        "created_at": created_at,
        "frozen": True,
        "target_language": "ko-KR",
        "training_allowed": False,
        "checkpoint_selection_allowed": False,
        "source_name": "AI Hub 한국어 방언 발화 데이터(경상도), Validation split",
        "source_version": "local archive; AI Hub dataSetSn=119",
        "license_or_access_policy": "AI Hub 데이터 이용정책; local evaluation only",
        "entries": entries,
    }
    return manifest, provenance


def _add_rows_to_exclusions(
    exclusions: dict[str, set[str]],
    rows: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
) -> None:
    by_id = {str(item["utterance_id"]): item for item in provenance}
    for row in rows:
        utterance_id = str(row["utterance_id"])
        exclusions["speakers"].add(str(row["speaker_id"]))
        exclusions["utterances"].add(utterance_id)
        exclusions["audio"].add(str(row["audio_sha256"]))
        exclusions["surfaces"].add(normalize_for_cer(str(row["text"])))
        if utterance_id in by_id:
            provenance_row = by_id[utterance_id]
            exclusions["recordings"].add(str(provenance_row["source_recording_id"]))
            exclusions["audio"].add(str(provenance_row["source_audio_sha256"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", required=True, type=Path)
    parser.add_argument("--source-tar", required=True, type=Path)
    parser.add_argument("--porter-dir", required=True, type=Path)
    parser.add_argument("--historical-train", required=True, type=Path)
    parser.add_argument("--historical-validation", required=True, type=Path)
    parser.add_argument("--historical-exclusions", required=True, type=Path)
    parser.add_argument("--consumed-test-v1", required=True, type=Path)
    parser.add_argument("--criteria", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    paths = {
        name: value.resolve(strict=True)
        for name, value in {
            "labels_dir": args.labels_dir,
            "source_tar": args.source_tar,
            "porter_dir": args.porter_dir,
            "historical_train": args.historical_train,
            "historical_validation": args.historical_validation,
            "historical_exclusions": args.historical_exclusions,
            "consumed_test_v1": args.consumed_test_v1,
            "criteria": args.criteria,
            "protocol": args.protocol,
        }.items()
    }
    porter = _load_porter(paths["porter_dir"])
    registry = _read_json(paths["historical_exclusions"])
    consumed_test = _read_json(paths["consumed_test_v1"])
    exclusions = _initial_exclusions(registry, consumed_test)
    expressions = _expression_index(paths["labels_dir"])
    pools, corpus = porter.load_speaker_pools(paths["labels_dir"], set(), set())
    pools = _filter_pools(pools, exclusions, expressions)
    allocations = _allocate(pools, exclusions, expressions)
    report = _allocation_report(allocations, expressions, corpus)
    if args.report_only:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required unless --report-only is used")

    output_dir = args.output_dir.resolve()
    staging = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists() or staging.exists():
        raise FileExistsError("output or staging directory already exists")
    staging.mkdir(parents=True)
    roots = {name: staging / name for name in ("train", "validation", "test-v2")}
    for root in roots.values():
        (root / "audio").mkdir(parents=True)
    source_dir = staging / "source-recordings"
    selected_pools = [pool for allocation in allocations for pool in allocation.speakers]
    source_inventory = porter.extract_recordings(
        paths["source_tar"],
        [pool.recording_id for pool in selected_pools],
        source_dir,
    )
    pool_by_key = {
        (pool.recording_id, pool.local_speaker_id): pool for pool in selected_pools
    }
    by_name = {allocation.spec.name: allocation for allocation in allocations}
    created_at = dt.datetime.now(dt.UTC).isoformat()

    historical_train = _safe_zip_manifest(
        paths["historical_train"],
        "train_manifest.jsonl",
        roots["train"],
        "train",
        porter,
    )
    historical_validation = _safe_zip_manifest(
        paths["historical_validation"],
        "validation_manifest.jsonl",
        roots["validation"],
        "validation",
        porter,
    )
    new_train, train_provenance = _new_development_rows(
        by_name["train"],
        roots["train"],
        source_inventory,
        pool_by_key,
        expressions,
        porter,
    )
    new_validation, validation_provenance = _new_development_rows(
        by_name["validation"],
        roots["validation"],
        source_inventory,
        pool_by_key,
        expressions,
        porter,
    )
    train_rows = historical_train + new_train
    validation_rows = historical_validation + new_validation
    _write_jsonl(roots["train"] / "manifest.source.jsonl", train_rows)
    _write_jsonl(
        roots["train"] / "manifest.absolute.jsonl",
        _absolute_rows(output_dir / "train", train_rows),
    )
    _write_jsonl(roots["validation"] / "manifest.source.jsonl", validation_rows)
    _write_jsonl(
        roots["validation"] / "manifest.absolute.jsonl",
        _absolute_rows(output_dir / "validation", validation_rows),
    )
    _write_json(roots["train"] / "new-selection-provenance.json", train_provenance)
    _write_json(
        roots["validation"] / "new-selection-provenance.json",
        validation_provenance,
    )

    test_manifest, test_provenance = _test_manifest(
        by_name["test-v2"],
        roots["test-v2"],
        source_inventory,
        pool_by_key,
        expressions,
        porter,
        created_at,
    )
    _write_json(roots["test-v2"] / "manifest.json", test_manifest)
    _write_json(roots["test-v2"] / "selection-provenance.json", test_provenance)

    final_exclusions = {name: set(values) for name, values in exclusions.items()}
    _add_rows_to_exclusions(final_exclusions, train_rows, train_provenance)
    _add_rows_to_exclusions(final_exclusions, validation_rows, validation_provenance)
    exclusion_manifest = {
        "schema_version": "1.0.0",
        "created_at": created_at,
        "source_artifacts": [
            *map(str, registry.get("source_artifacts", [])),
            f"{paths['consumed_test_v1']}#sha256:{porter.sha256_file(paths['consumed_test_v1'])}",
            f"{output_dir / 'train' / 'manifest.source.jsonl'}",
            f"{output_dir / 'validation' / 'manifest.source.jsonl'}",
        ],
        "speaker_ids": sorted(final_exclusions["speakers"]),
        "utterance_ids": sorted(final_exclusions["utterances"]),
        "source_recording_ids": sorted(final_exclusions["recordings"]),
        "audio_sha256s": sorted(final_exclusions["audio"]),
        "normalized_surface_texts": sorted(final_exclusions["surfaces"]),
    }
    _write_json(staging / "test-v2-exclusions.json", exclusion_manifest)
    for source_info in source_inventory["recordings"].values():
        extracted_name = Path(str(source_info["extracted_path"])).name
        source_info["extracted_path"] = str(
            (output_dir / "source-recordings" / extracted_name).resolve()
        )
    _write_json(staging / "source-inventory.json", source_inventory)

    identities = {
        name: {
            "speakers": {row["speaker_id"] for row in rows},
            "utterances": {row["utterance_id"] for row in rows},
            "audio": {row["audio_sha256"] for row in rows},
            "surfaces": {normalize_for_cer(row["text"]) for row in rows},
        }
        for name, rows in {"train": train_rows, "validation": validation_rows}.items()
    }
    identities["test-v2"] = {
        "speakers": {row["speaker_id"] for row in test_manifest["entries"]},
        "utterances": {row["utterance_id"] for row in test_manifest["entries"]},
        "audio": {row["audio_sha256"] for row in test_manifest["entries"]},
        "surfaces": {
            normalize_for_cer(row["surface_text"]) for row in test_manifest["entries"]
        },
    }
    overlaps: dict[str, dict[str, list[str]]] = {}
    for left, right in (("train", "validation"), ("train", "test-v2"), ("validation", "test-v2")):
        overlaps[f"{left}__{right}"] = {
            field: sorted(identities[left][field] & identities[right][field])
            for field in identities[left]
        }
    failures = {
        "train_count": len(train_rows) != 1_000,
        "train_speakers": len(identities["train"]["speakers"]) != 41,
        "validation_count": len(validation_rows) != 140,
        "validation_speakers": len(identities["validation"]["speakers"]) != 14,
        "test_count": len(test_manifest["entries"]) != 100,
        "test_speakers": len(identities["test-v2"]["speakers"]) != 10,
        "cross_split_overlap": any(
            values for pair in overlaps.values() for values in pair.values()
        ),
    }
    if any(failures.values()):
        raise ValueError(f"final allocation checks failed: {failures}; overlaps={overlaps}")

    commitment = {
        "schema_version": "1.0.0",
        "status": "frozen_before_training",
        "created_at": created_at,
        "one_attempt": True,
        "protocol_path": str(paths["protocol"]),
        "protocol_sha256": porter.sha256_file(paths["protocol"]),
        "criteria_path": str(paths["criteria"]),
        "criteria_sha256": porter.sha256_file(paths["criteria"]),
        "train_manifest_sha256": porter.sha256_file(
            roots["train"] / "manifest.source.jsonl"
        ),
        "validation_manifest_sha256": porter.sha256_file(
            roots["validation"] / "manifest.source.jsonl"
        ),
        "test_v2_manifest_sha256": porter.sha256_file(
            roots["test-v2"] / "manifest.json"
        ),
        "counts": {
            "train_utterances": len(train_rows),
            "train_speakers": len(identities["train"]["speakers"]),
            "validation_utterances": len(validation_rows),
            "validation_speakers": len(identities["validation"]["speakers"]),
            "test_v2_utterances": len(test_manifest["entries"]),
            "test_v2_speakers": len(identities["test-v2"]["speakers"]),
        },
        "overlaps": overlaps,
        "selection": report,
        "failures": failures,
    }
    _write_json(staging / "commitment.json", commitment)
    staging.rename(output_dir)
    print(f"output_dir={output_dir}")
    print(json.dumps(commitment["counts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
