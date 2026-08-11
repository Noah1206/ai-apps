#!/usr/bin/env python3
"""Validate one append-only TASK-003C diagnostic result directory."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from contract import ContractError
from run_diagnostics import DIAGNOSTIC_TARGETS, OUTPUT_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = validate_diagnostics(args.diagnostics_dir)
    except (ContractError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        "TASK-003C diagnostics valid: "
        f"{summary['captured_target_count']}/{summary['expected_target_count']} targets"
    )
    return 0


def validate_diagnostics(diagnostics_dir: Path) -> Mapping[str, Any]:
    root = diagnostics_dir.expanduser().resolve()
    if not root.is_dir():
        raise ContractError(f"diagnostics directory is missing: {root}")
    missing_files = [name for name in OUTPUT_NAMES if not (root / name).is_file()]
    if missing_files:
        raise ContractError(f"diagnostic files are missing: {missing_files}")
    if not (root / "run.log").read_text(encoding="utf-8").strip():
        raise ContractError("run.log is empty")

    audio_probe = _load_object(root / "audio_probe.json")
    raw_documents = _load_jsonl(root / "raw_model_output.jsonl")
    adapter_documents = _load_jsonl(root / "adapter_trace.jsonl")
    summary = _load_object(root / "task_003c_summary.json")
    expected_ids = {
        str(target["utterance_id"])
        for target in DIAGNOSTIC_TARGETS
    }

    _require_task(audio_probe, "audio_probe.json")
    _require_task(summary, "task_003c_summary.json")
    probes = _require_list(audio_probe, "targets", "audio_probe.json")
    _validate_record_set(probes, expected_ids, location="audio_probe.json.targets")
    _validate_record_set(
        raw_documents,
        expected_ids,
        location="raw_model_output.jsonl",
    )
    _validate_record_set(
        adapter_documents,
        expected_ids,
        location="adapter_trace.jsonl",
    )

    for index, probe in enumerate(probes, start=1):
        location = f"audio_probe.json.targets[{index}]"
        if probe.get("sample_rate_hz") != 16_000:
            raise ContractError(f"{location} is not 16000 Hz")
        if probe.get("channels") != 1:
            raise ContractError(f"{location} is not mono")
        if probe.get("sample_width_bits") != 16 or probe.get("compression") != "NONE":
            raise ContractError(f"{location} is not uncompressed PCM16")
        if probe.get("hash_matches_manifest") is not True:
            raise ContractError(f"{location} hash does not match the manifest")
        if not isinstance(probe.get("sample_count"), int) or probe["sample_count"] <= 0:
            raise ContractError(f"{location} has no waveform samples")
        if probe.get("non_finite_sample_count") != 0:
            raise ContractError(f"{location} reports non-finite samples")

    for index, document in enumerate(raw_documents, start=1):
        _require_task(document, f"raw_model_output.jsonl line {index}")
        raw_output = document.get("raw_model_output")
        if not isinstance(raw_output, dict):
            raise ContractError(
                f"raw_model_output.jsonl line {index}.raw_model_output must be an object"
            )

    for index, document in enumerate(adapter_documents, start=1):
        location = f"adapter_trace.jsonl line {index}"
        _require_task(document, location)
        processor_call = document.get("processor_call")
        generate_call = document.get("model_generate_call")
        if not isinstance(processor_call, dict) or processor_call.get("language") != "ko-KR":
            raise ContractError(f"{location} did not record processor language ko-KR")
        if processor_call.get("is_streaming_passed") is not False:
            raise ContractError(f"{location} unexpectedly passed streaming mode")
        if not isinstance(generate_call, dict) or generate_call.get("streamer_passed") is not False:
            raise ContractError(f"{location} unexpectedly passed a streamer")
        transcript = document.get("adapter_transcript")
        if transcript is not None and not isinstance(transcript, str):
            raise ContractError(f"{location}.adapter_transcript must be string or null")

    if summary.get("status") != "complete":
        raise ContractError("task_003c_summary.json status is not complete")
    if summary.get("expected_target_count") != len(expected_ids):
        raise ContractError("task_003c_summary.json expected_target_count is invalid")
    if summary.get("captured_target_count") != len(expected_ids):
        raise ContractError("task_003c_summary.json captured_target_count is invalid")
    if summary.get("missing_utterance_ids") != []:
        raise ContractError("task_003c_summary.json contains missing utterance IDs")
    if summary.get("failures") != []:
        raise ContractError("task_003c_summary.json contains failures")
    assessment = summary.get("diagnostic_assessment")
    allowed_assessments = {
        "MODEL_RETURNED_EMPTY",
        "ADAPTER_EXTRACTION_ERROR",
        "AUDIO_CONTRACT_ERROR",
        "INFERENCE_CONFIGURATION_ERROR",
        "NOT_REPRODUCED",
        "UNRESOLVED",
    }
    if (
        not isinstance(assessment, dict)
        or assessment.get("classification") not in allowed_assessments
        or not isinstance(assessment.get("evidence"), list)
    ):
        raise ContractError("task_003c_summary.json diagnostic_assessment is invalid")
    summary_targets = _require_list(summary, "targets", "task_003c_summary.json")
    _validate_record_set(
        summary_targets,
        expected_ids,
        location="task_003c_summary.json.targets",
    )
    return summary


def _load_object(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ContractError(f"{path.name} is not valid JSON") from error
    if not isinstance(document, dict):
        raise ContractError(f"{path.name} must contain one JSON object")
    return document


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            raise ContractError(f"{path.name} contains a blank line at {line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ContractError(
                f"{path.name} contains invalid JSON at line {line_number}"
            ) from error
        if not isinstance(record, dict):
            raise ContractError(f"{path.name} line {line_number} is not an object")
        records.append(record)
    if not records:
        raise ContractError(f"{path.name} is empty")
    return records


def _require_task(document: Mapping[str, Any], location: str) -> None:
    if document.get("task_id") != "TASK-003C":
        raise ContractError(f"{location} is not a TASK-003C document")


def _require_list(
    document: Mapping[str, Any],
    key: str,
    location: str,
) -> list[dict[str, Any]]:
    value = document.get(key)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ContractError(f"{location}.{key} must be an array of objects")
    return value


def _validate_record_set(
    records: list[dict[str, Any]],
    expected_ids: set[str],
    *,
    location: str,
) -> None:
    ids: list[str] = []
    for record in records:
        utterance_id = record.get("utterance_id")
        if not isinstance(utterance_id, str):
            raise ContractError(f"{location} contains an invalid utterance ID")
        ids.append(utterance_id)
    if len(ids) != len(set(ids)):
        raise ContractError(f"{location} contains duplicate utterance IDs")
    actual_ids = set(ids)
    if actual_ids != expected_ids:
        raise ContractError(
            f"{location} target IDs differ: "
            f"missing={sorted(expected_ids - actual_ids)}, "
            f"unexpected={sorted(actual_ids - expected_ids)}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
