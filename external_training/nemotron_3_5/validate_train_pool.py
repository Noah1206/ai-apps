#!/usr/bin/env python3
"""Validate a TASK-005 train-only handoff ZIP using the Python standard library."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import wave
import zipfile
from pathlib import Path, PurePosixPath

SHA256 = re.compile(r"^[0-9a-f]{64}$")
ROW_FIELDS = {
    "audio_filepath",
    "duration",
    "text",
    "target_lang",
    "speaker_id",
    "utterance_id",
    "audio_sha256",
    "split",
}


def _safe_name(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"unsafe ZIP member: {name}")


def validate(package: Path) -> dict[str, object]:
    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("ZIP contains duplicate member names")
        for name in names:
            _safe_name(name)
        required = {
            "package_metadata.json",
            "train_manifest.jsonl",
            "VALIDATION_REQUIRED.txt",
        }
        if not required.issubset(names):
            raise ValueError(f"missing required files: {sorted(required - set(names))}")
        unexpected = {
            name for name in names if name not in required and not name.startswith("audio/")
        }
        if unexpected:
            raise ValueError(f"unexpected package files: {sorted(unexpected)}")

        metadata = json.loads(archive.read("package_metadata.json"))
        if metadata.get("status") != "train_pool_ready_validation_missing":
            raise ValueError("unexpected package status")
        if metadata.get("training_permitted") is not False:
            raise ValueError("train-only package must not permit training")
        if metadata.get("validation_required") is not True:
            raise ValueError("train-only package must require validation")
        if metadata.get("target_language") != "ko-KR":
            raise ValueError("target_language must be ko-KR")

        rows = [
            json.loads(line)
            for line in archive.read("train_manifest.jsonl").decode("utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != metadata.get("train_utterance_count"):
            raise ValueError("manifest count does not match package metadata")
        if metadata.get("validation_utterance_count") != 0:
            raise ValueError("train-only package must contain zero validation utterances")

        utterance_ids: set[str] = set()
        audio_paths: set[str] = set()
        audio_hashes: set[str] = set()
        total_duration = 0.0
        for row in rows:
            if set(row) != ROW_FIELDS:
                raise ValueError("manifest row has missing or unknown fields")
            if row["target_lang"] != "ko-KR" or row["split"] != "train":
                raise ValueError("manifest row has an invalid language or split")
            if not isinstance(row["text"], str) or not row["text"].strip():
                raise ValueError("manifest row has an empty transcript")
            if not isinstance(row["duration"], (int, float)) or row["duration"] <= 0:
                raise ValueError("manifest row has an invalid duration")
            if not SHA256.fullmatch(row["audio_sha256"]):
                raise ValueError("manifest row has an invalid audio SHA-256")
            audio_path = row["audio_filepath"]
            _safe_name(audio_path)
            if not audio_path.startswith("audio/") or not audio_path.endswith(".wav"):
                raise ValueError("manifest audio path must be audio/<id>.wav")
            if audio_path not in names:
                raise ValueError(f"manifest audio is missing: {audio_path}")
            if (
                row["utterance_id"] in utterance_ids
                or audio_path in audio_paths
                or row["audio_sha256"] in audio_hashes
            ):
                raise ValueError("manifest contains a duplicate utterance or audio")
            utterance_ids.add(row["utterance_id"])
            audio_paths.add(audio_path)
            audio_hashes.add(row["audio_sha256"])

            payload = archive.read(audio_path)
            if hashlib.sha256(payload).hexdigest() != row["audio_sha256"]:
                raise ValueError(f"audio hash mismatch: {audio_path}")
            with archive.open(audio_path) as stream, wave.open(stream) as wav:
                if (
                    wav.getframerate() != 16000
                    or wav.getnchannels() != 1
                    or wav.getsampwidth() != 2
                    or wav.getnframes() <= 0
                ):
                    raise ValueError(f"audio contract mismatch: {audio_path}")
                measured_duration = wav.getnframes() / wav.getframerate()
            if abs(measured_duration - row["duration"]) > 0.02:
                raise ValueError(f"audio duration mismatch: {audio_path}")
            total_duration += row["duration"]

        packaged_audio = {name for name in names if name.startswith("audio/")}
        if packaged_audio != audio_paths:
            raise ValueError("ZIP audio members do not exactly match the manifest")
        if abs(total_duration - metadata["total_duration_seconds"]) > 0.02:
            raise ValueError("total duration does not match package metadata")
        return {
            "status": "valid_train_pool",
            "train_utterances": len(rows),
            "speaker_id": metadata["speaker_id"],
            "target_language": metadata["target_language"],
            "validation_required": True,
            "training_permitted": False,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.package), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
