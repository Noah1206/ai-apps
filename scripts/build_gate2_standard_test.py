#!/usr/bin/env python3
"""Build a Gate 2 Standard Korean regression set from Zeroth-Korean test data."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import subprocess
import wave
from pathlib import Path
from typing import Any

from busan_lab.evaluation.metrics import normalize_for_cer

ZEROTH_TEST_PARQUET_SHA256 = (
    "2f6382b902622ede1bf35e5a167a9b06b0f1b8c896e1947324b8b5180fca7502"
)


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _clean_text(text: str) -> bool:
    normalized = normalize_for_cer(text)
    return bool(normalized) and len(normalized) >= 8 and len(text.split()) >= 3


def _choose_evenly(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"only {len(rows)} eligible rows; {count} required")
    indexes = [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)]
    if len(set(indexes)) != count:
        raise AssertionError("even selection produced duplicate indexes")
    return [rows[index] for index in indexes]


def _convert_audio(audio: bytes, destination: Path) -> None:
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-map_metadata",
        "-1",
        "-map",
        "0:a:0",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        "-fflags",
        "+bitexact",
        "-flags:a",
        "+bitexact",
        "-n",
        str(destination),
    ]
    subprocess.run(command, input=audio, check=True)


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        contract = (
            wav.getframerate(),
            wav.getnchannels(),
            wav.getsampwidth(),
            wav.getcomptype(),
        )
        if contract != (16_000, 1, 2, "NONE") or wav.getnframes() <= 0:
            raise ValueError(f"invalid output WAV contract: {path}: {contract}")
        payload = wav.readframes(wav.getnframes())
        if not payload or not any(payload):
            raise ValueError(f"empty or silent WAV: {path}")
        return wav.getnframes() / wav.getframerate()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--exclusions", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--speaker-count", type=int, default=10)
    parser.add_argument("--utterances-per-speaker", type=int, default=10)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("install pyarrow or run with: uv run --with pyarrow") from error

    parquet_path = args.parquet.resolve(strict=True)
    exclusion_path = args.exclusions.resolve(strict=True)
    parquet_sha256 = _hash_file(parquet_path)
    if parquet_sha256 != ZEROTH_TEST_PARQUET_SHA256:
        raise ValueError(
            "Zeroth-Korean test Parquet SHA-256 mismatch: "
            f"expected {ZEROTH_TEST_PARQUET_SHA256}, got {parquet_sha256}"
        )
    exclusions = _read_json(exclusion_path)
    excluded_speakers = set(map(str, exclusions.get("speaker_ids", [])))
    excluded_utterances = set(map(str, exclusions.get("utterance_ids", [])))
    excluded_recordings = set(map(str, exclusions.get("source_recording_ids", [])))
    excluded_surfaces = set(map(str, exclusions.get("normalized_surface_texts", [])))

    table = pq.read_table(parquet_path)
    required_columns = {"id", "speaker_id", "chapter_id", "path", "audio", "text"}
    missing_columns = required_columns - set(table.column_names)
    if missing_columns:
        raise ValueError(f"missing Parquet columns: {sorted(missing_columns)}")
    rows_by_speaker: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row_index, row in enumerate(table.to_pylist()):
        utterance_id = f"zeroth-korean-test:{row['id']}"
        speaker_id = f"zeroth-korean-test-speaker-{row['speaker_id']}"
        source_recording_id = utterance_id
        text = str(row["text"]).strip()
        if (
            speaker_id in excluded_speakers
            or utterance_id in excluded_utterances
            or source_recording_id in excluded_recordings
            or normalize_for_cer(text) in excluded_surfaces
            or not _clean_text(text)
        ):
            continue
        row["_row_index"] = row_index
        rows_by_speaker[speaker_id].append(row)

    eligible_speakers = sorted(
        (
            (speaker_id, rows)
            for speaker_id, rows in rows_by_speaker.items()
            if len(rows) >= args.utterances_per_speaker
        ),
        key=lambda item: item[0],
    )
    if len(eligible_speakers) < args.speaker_count:
        raise ValueError(
            f"only {len(eligible_speakers)} speakers have at least "
            f"{args.utterances_per_speaker} eligible utterances"
        )
    selected = [
        (speaker_id, row)
        for speaker_id, rows in eligible_speakers[: args.speaker_count]
        for row in _choose_evenly(rows, args.utterances_per_speaker)
    ]
    summary = {
        "source_rows": table.num_rows,
        "source_speakers": len(rows_by_speaker),
        "eligible_speakers": len(eligible_speakers),
        "selected_utterances": len(selected),
        "selected_speakers": args.speaker_count,
        "utterances_per_speaker": args.utterances_per_speaker,
        "speaker_counts": dict(collections.Counter(item[0] for item in selected)),
    }
    if args.report_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required unless --report-only is used")

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    audio_dir = output_dir / "audio"
    output_dir.mkdir(parents=True)
    audio_dir.mkdir()
    entries: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for speaker_id, row in selected:
        source_id = str(row["id"])
        utterance_id = f"zeroth-korean-test:{source_id}"
        source_recording_id = utterance_id
        audio_value = row["audio"]
        if not isinstance(audio_value, dict) or not isinstance(audio_value.get("bytes"), bytes):
            raise ValueError(f"missing embedded audio bytes: {source_id}")
        source_audio = audio_value["bytes"]
        safe_source_id = source_id.replace("/", "_").replace("\\", "_")
        audio_path = audio_dir / f"{safe_source_id}.wav"
        _convert_audio(source_audio, audio_path)
        duration = _wav_duration(audio_path)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"invalid duration: {source_id}")
        entries.append(
            {
                "utterance_id": utterance_id,
                "speaker_id": speaker_id,
                "source_recording_id": source_recording_id,
                "region": "대한민국 (표준 한국어 읽기 발화)",
                "audio_filepath": f"audio/{audio_path.name}",
                "audio_sha256": _hash_file(audio_path),
                "audio_lineage_sha256s": [_hash_bytes(source_audio)],
                "duration_seconds": duration,
                "surface_text": str(row["text"]).strip(),
                "dialect_expressions": [],
                "label_status": "human_reviewed",
            }
        )
        provenance.append(
            {
                "utterance_id": utterance_id,
                "parquet_row_index": row["_row_index"],
                "zeroth_id": source_id,
                "zeroth_speaker_id": row["speaker_id"],
                "zeroth_chapter_id": row["chapter_id"],
                "zeroth_original_path": row["path"],
                "embedded_audio_path": audio_value.get("path"),
            }
        )

    now = dt.datetime.now(dt.UTC).isoformat()
    manifest = {
        "schema_version": "1.0.0",
        "dataset_id": "zeroth-korean-standard-regression",
        "dataset_version": "1.0.0",
        "dataset_kind": "standard_korean_regression",
        "created_at": now,
        "frozen": True,
        "target_language": "ko-KR",
        "training_allowed": False,
        "checkpoint_selection_allowed": False,
        "source_name": "Zeroth-Korean (OpenSLR SLR40), test split",
        "source_version": "kresnik/zeroth_korean Parquet conversion 2.0.0",
        "license_or_access_policy": "CC BY 4.0; attribution required",
        "entries": entries,
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "selection-provenance.json", provenance)
    _write_json(
        output_dir / "build-report.json",
        {
            **summary,
            "created_at": now,
            "source_parquet": str(parquet_path),
            "source_parquet_sha256": parquet_sha256,
            "source_parquet_bytes": parquet_path.stat().st_size,
            "source_repository": "https://huggingface.co/datasets/kresnik/zeroth_korean",
            "canonical_source": "https://www.openslr.org/40/",
            "license": "CC BY 4.0",
            "label_status_basis": (
                "OpenSLR describes the release as transcribed Korean audio; the selected "
                "test references are source-provided rather than model-generated"
            ),
            "exclusion_registry": str(exclusion_path),
            "exclusion_registry_sha256": _hash_file(exclusion_path),
            "duration_seconds": sum(entry["duration_seconds"] for entry in entries),
        },
    )
    print(f"manifest={output_dir / 'manifest.json'}")
    print(f"utterances={len(entries)}")
    print(f"speakers={len({entry['speaker_id'] for entry in entries})}")
    print(f"duration_seconds={sum(entry['duration_seconds'] for entry in entries):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
