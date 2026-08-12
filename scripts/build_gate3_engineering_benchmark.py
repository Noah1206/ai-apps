"""Build a deterministic, transcript-free Gate 3 engineering benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_SEED = "gate3-engineering-v1-2026-08-12"
DEFAULT_BENCHMARK_ID = "gate3-streaming-engineering-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speaker-count", type=int, default=10)
    parser.add_argument("--samples-per-speaker", type=int, default=2)
    parser.add_argument("--min-duration-seconds", type=float, default=2.5)
    parser.add_argument("--max-duration-seconds", type=float, default=7.0)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--benchmark-id", default=DEFAULT_BENCHMARK_ID)
    parser.add_argument(
        "--exclude-manifest",
        action="append",
        type=Path,
        default=[],
        help="Exclude utterance IDs and audio hashes already used by another benchmark.",
    )
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selection_key(seed: str, kind: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{kind}\0{value}".encode()).hexdigest()


def speaker_commitment(speaker_id: str) -> str:
    return hashlib.sha256(f"gate3-speaker\0{speaker_id}".encode()).hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {
                "utterance_id",
                "audio_filepath",
                "duration",
                "speaker_id",
                "audio_sha256",
            }
            missing = required - row.keys()
            if missing:
                raise ValueError(f"source line {line_number} is missing {sorted(missing)}")
            rows.append(row)
    return rows


def exclusion_keys(paths: list[Path]) -> tuple[set[str], set[str]]:
    utterance_ids: set[str] = set()
    audio_sha256s: set[str] = set()
    for path in paths:
        with path.resolve().open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                utterance_id = row.get("utterance_id")
                audio_sha256 = row.get("audio_sha256")
                if not isinstance(utterance_id, str) or not isinstance(audio_sha256, str):
                    raise ValueError(f"exclude line {line_number} in {path} is invalid")
                utterance_ids.add(utterance_id)
                audio_sha256s.add(audio_sha256)
    return utterance_ids, audio_sha256s


def select_rows(
    rows: list[dict[str, Any]],
    *,
    seed: str,
    speaker_count: int,
    samples_per_speaker: int,
    min_duration_seconds: float,
    max_duration_seconds: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        duration = float(row["duration"])
        if min_duration_seconds <= duration <= max_duration_seconds:
            grouped[str(row["speaker_id"])].append(row)
    eligible = [
        speaker_id for speaker_id, samples in grouped.items() if len(samples) >= samples_per_speaker
    ]
    ranked_speakers = sorted(
        eligible,
        key=lambda value: selection_key(seed, "speaker", value),
    )
    if len(ranked_speakers) < speaker_count:
        raise ValueError("not enough eligible speakers for the requested benchmark")
    selected: list[dict[str, Any]] = []
    for speaker_id in ranked_speakers[:speaker_count]:
        ranked_samples = sorted(
            grouped[speaker_id],
            key=lambda row: selection_key(seed, "utterance", str(row["utterance_id"])),
        )
        selected.extend(ranked_samples[:samples_per_speaker])
    return selected


def build(arguments: argparse.Namespace) -> dict[str, Any]:
    source_manifest = arguments.source_manifest.resolve()
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if arguments.speaker_count < 5 or arguments.samples_per_speaker < 1:
        raise ValueError("Gate 3 benchmark requires at least five speakers")
    excluded_utterance_ids, excluded_audio_sha256s = exclusion_keys(arguments.exclude_manifest)
    source_rows = [
        row
        for row in load_rows(source_manifest)
        if str(row["utterance_id"]) not in excluded_utterance_ids
        and str(row["audio_sha256"]) not in excluded_audio_sha256s
    ]
    rows = select_rows(
        source_rows,
        seed=arguments.seed,
        speaker_count=arguments.speaker_count,
        samples_per_speaker=arguments.samples_per_speaker,
        min_duration_seconds=arguments.min_duration_seconds,
        max_duration_seconds=arguments.max_duration_seconds,
    )
    benchmark_rows = []
    for row in rows:
        audio_path = Path(str(row["audio_filepath"]))
        if not audio_path.is_absolute():
            audio_path = source_manifest.parent / audio_path
        audio_path = audio_path.resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        observed_sha256 = sha256_file(audio_path)
        if observed_sha256 != row["audio_sha256"]:
            raise ValueError(f"audio hash mismatch: {row['utterance_id']}")
        benchmark_rows.append(
            {
                "case_id": f"gate3-{len(benchmark_rows) + 1:03d}",
                "utterance_id": row["utterance_id"],
                "speaker_id": row["speaker_id"],
                "audio_filepath": str(audio_path),
                "audio_sha256": observed_sha256,
                "duration_seconds": float(row["duration"]),
                "source_split": str(row.get("split", "validation")),
            }
        )
    manifest_path = output_dir / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in benchmark_rows),
        encoding="utf-8",
    )
    commitment = {
        "schema_version": "1.0.0",
        "benchmark_id": arguments.benchmark_id,
        "status": "frozen_before_streaming_outputs",
        "source_manifest_sha256": sha256_file(source_manifest),
        "manifest_sha256": sha256_file(manifest_path),
        "selection_seed": arguments.seed,
        "selection_algorithm": (
            "exclude_prior_utterance_and_audio_hashes_then_duration_filter_then_"
            "sha256_rank_speakers_then_sha256_rank_utterances"
        ),
        "excluded_manifest_sha256s": [
            sha256_file(path.resolve()) for path in arguments.exclude_manifest
        ],
        "excluded_utterance_count": len(excluded_utterance_ids),
        "excluded_audio_sha256_count": len(excluded_audio_sha256s),
        "duration_range_seconds": [
            arguments.min_duration_seconds,
            arguments.max_duration_seconds,
        ],
        "case_count": len(benchmark_rows),
        "speaker_count": len({row["speaker_id"] for row in benchmark_rows}),
        "samples_per_speaker": arguments.samples_per_speaker,
        "source_role": "checkpoint_validation_reused_for_streaming_engineering_only",
        "quality_claim_allowed": False,
        "cases": [
            {
                "case_id": row["case_id"],
                "audio_sha256": row["audio_sha256"],
                "speaker_commitment": speaker_commitment(str(row["speaker_id"])),
                "duration_seconds": row["duration_seconds"],
            }
            for row in benchmark_rows
        ],
    }
    (output_dir / "commitment.json").write_text(
        json.dumps(commitment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return commitment


def main() -> int:
    result = build(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
