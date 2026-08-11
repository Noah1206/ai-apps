#!/usr/bin/env python3
"""Capture TASK-003C raw Nemotron output and Adapter traces for four fixed audios."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import logging
import math
import os
import sys
import tempfile
import time
import wave
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adapter import (
    SpeechModelAdapter,
    TranscriptionTrace,
    TransformersNemotronAdapter,
)
from contract import (
    ContractError,
    ValidatedBenchmark,
    extract_validated_audio,
)
from run_inference import load_config, validate_from_config

DIAGNOSTIC_TARGETS = (
    {
        "utterance_id": "3faa344e-968d-42cb-baf8-f1847a936a98",
        "role": "baseline_empty_output",
    },
    {
        "utterance_id": "75b96c24-cab0-4e52-839b-dd67b496e58a",
        "role": "baseline_empty_output",
    },
    {
        "utterance_id": "c95d2f85-a98a-4c19-9476-e55b92a49dc0",
        "role": "baseline_empty_output",
    },
    {
        "utterance_id": "38f63a59-325b-4e53-9287-ab094c6a889d",
        "role": "baseline_nonempty_control",
    },
)
OUTPUT_NAMES = (
    "audio_probe.json",
    "raw_model_output.jsonl",
    "adapter_trace.jsonl",
    "task_003c_summary.json",
    "run.log",
)


class DiagnosticRunError(RuntimeError):
    """Raised after preserving all available TASK-003C diagnostic evidence."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-package", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        benchmark = validate_from_config(args.benchmark_package, config)
        run_diagnostics(
            benchmark_package=args.benchmark_package,
            benchmark=benchmark,
            config=config,
            output_dir=args.output_dir,
        )
    except (ContractError, DiagnosticRunError, OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


def run_diagnostics(
    *,
    benchmark_package: Path,
    benchmark: ValidatedBenchmark,
    config: Mapping[str, Any],
    output_dir: Path,
    adapter_factory: Callable[..., SpeechModelAdapter] = TransformersNemotronAdapter,
) -> None:
    output_dir = output_dir.expanduser().resolve()
    _prepare_output_directory(output_dir)
    logger = _create_logger(output_dir / "run.log")
    started_at = _utc_now()
    targets = _target_entries(benchmark)
    model = _mapping(config, "model")
    runtime = _mapping(config, "runtime")

    raw_documents: list[dict[str, Any]] = []
    adapter_documents: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    target_summaries: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="task-003c-audio-") as temporary:
        audio_by_id = extract_validated_audio(
            benchmark_package,
            benchmark,
            Path(temporary),
        )
        probes = [
            _audio_probe(
                audio_by_id[target["entry"].utterance_id],
                target["entry"],
                role=target["role"],
            )
            for target in targets
        ]
        _write_new_json(
            output_dir / "audio_probe.json",
            {
                "schema_version": "1.0.0",
                "task_id": "TASK-003C",
                "benchmark_id": benchmark.benchmark_id,
                "benchmark_version": benchmark.benchmark_version,
                "benchmark_package_sha256": benchmark.package_sha256,
                "targets": probes,
            },
        )
        logger.info("audio contract captured for %d fixed targets", len(probes))

        cache_dir_raw = runtime.get("cache_dir")
        cache_dir = Path(cache_dir_raw) if isinstance(cache_dir_raw, str) else None
        adapter = adapter_factory(
            model_id=_string(model, "id"),
            revision=_string(model, "revision"),
            target_language=_string(model, "target_language"),
            cache_dir=cache_dir,
        )
        warmup_ms = adapter.warmup()
        logger.info("synthetic warm-up completed in %.2f ms", warmup_ms)

        for target in targets:
            entry = target["entry"]
            try:
                adapter.synchronize()
                inference_started = time.perf_counter()
                trace = adapter.transcribe_with_trace(audio_by_id[entry.utterance_id])
                adapter.synchronize()
                latency_ms = (time.perf_counter() - inference_started) * 1000
                raw_documents.append(
                    _raw_output_document(
                        entry=entry,
                        role=target["role"],
                        trace=trace,
                        adapter=adapter,
                    )
                )
                adapter_documents.append(
                    _adapter_trace_document(
                        entry=entry,
                        role=target["role"],
                        trace=trace,
                        latency_ms=latency_ms,
                    )
                )
                target_summaries.append(
                    _target_summary(
                        entry=entry,
                        role=target["role"],
                        trace=trace,
                        latency_ms=latency_ms,
                    )
                )
                if trace.adapter_transcript is None:
                    failures.append(
                        {
                            "utterance_id": entry.utterance_id,
                            "error_type": "AdapterExtractionError",
                            "message": _trace_error_message(trace),
                        }
                    )
                logger.info(
                    "captured raw output and Adapter trace for %s in %.2f ms",
                    entry.utterance_id,
                    latency_ms,
                )
            except Exception as error:
                failures.append(
                    {
                        "utterance_id": entry.utterance_id,
                        "error_type": type(error).__name__,
                        "message": _sanitize_error_message(str(error)),
                    }
                )
                logger.exception("diagnostic inference failed for %s", entry.utterance_id)

    _write_jsonl(output_dir / "raw_model_output.jsonl", raw_documents)
    _write_jsonl(output_dir / "adapter_trace.jsonl", adapter_documents)
    completed_at = _utc_now()
    captured_ids = {item["utterance_id"] for item in target_summaries}
    expected_ids = {target["entry"].utterance_id for target in targets}
    status = (
        "complete"
        if captured_ids == expected_ids and not failures
        else "incomplete"
    )
    _write_new_json(
        output_dir / "task_003c_summary.json",
        {
            "schema_version": "1.0.0",
            "task_id": "TASK-003C",
            "status": status,
            "started_at": started_at,
            "completed_at": completed_at,
            "benchmark_id": benchmark.benchmark_id,
            "benchmark_version": benchmark.benchmark_version,
            "benchmark_package_sha256": benchmark.package_sha256,
            "model_id": adapter.model_id,
            "requested_revision": adapter.requested_revision,
            "resolved_revision": adapter.resolved_revision,
            "model_artifact_hash": f"sha256:{adapter.model_artifact_sha256}",
            "target_language": _string(model, "target_language"),
            "offline_inference": {
                "api": "model.generate",
                "per_utterance": True,
                "is_streaming_passed": False,
                "streamer_passed": False,
                "chunk_generator_passed": False,
            },
            "warmup_ms": round(warmup_ms, 3),
            "expected_target_count": len(expected_ids),
            "captured_target_count": len(captured_ids),
            "missing_utterance_ids": sorted(expected_ids - captured_ids),
            "failures": failures,
            "targets": target_summaries,
            "diagnostic_assessment": _assess_diagnostics(
                probes=probes,
                targets=target_summaries,
            ),
        },
    )
    if status != "complete":
        raise DiagnosticRunError(
            "TASK-003C diagnostics are incomplete; partial evidence was preserved"
        )


def _target_entries(benchmark: ValidatedBenchmark) -> list[dict[str, Any]]:
    entries = {entry.utterance_id: entry for entry in benchmark.entries}
    expected_ids = {target["utterance_id"] for target in DIAGNOSTIC_TARGETS}
    missing = sorted(expected_ids - set(entries))
    if missing:
        raise ContractError(f"TASK-003C target utterances are missing: {missing}")
    return [
        {
            "entry": entries[target["utterance_id"]],
            "role": target["role"],
        }
        for target in DIAGNOSTIC_TARGETS
    ]


def _audio_probe(path: Path, entry: Any, *, role: str) -> dict[str, Any]:
    with wave.open(str(path), "rb") as audio:
        channels = audio.getnchannels()
        sample_rate = audio.getframerate()
        sample_width_bytes = audio.getsampwidth()
        frame_count = audio.getnframes()
        compression = audio.getcomptype()
        frames = audio.readframes(frame_count)
    samples = array.array("h")
    samples.frombytes(frames)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        raise ContractError(f"TASK-003C audio has no samples: {entry.utterance_id}")
    normalized = [sample / 32768.0 for sample in samples]
    peak = max(abs(sample) for sample in normalized)
    rms = math.sqrt(sum(sample * sample for sample in normalized) / len(normalized))
    dc_offset = sum(normalized) / len(normalized)
    nonzero_ratio = sum(sample != 0 for sample in samples) / len(samples)
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "utterance_id": entry.utterance_id,
        "role": role,
        "archive_path": entry.archive_path,
        "manifest_audio_sha256": entry.audio_sha256,
        "audio_sha256": actual_hash,
        "hash_matches_manifest": actual_hash == entry.audio_sha256,
        "file_size_bytes": path.stat().st_size,
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "sample_width_bits": sample_width_bytes * 8,
        "compression": compression,
        "frame_count": frame_count,
        "sample_count": len(samples),
        "non_finite_sample_count": 0,
        "duration_ms": round(frame_count / sample_rate * 1000, 3),
        "peak_amplitude": round(peak, 8),
        "rms_amplitude": round(rms, 8),
        "dc_offset": round(dc_offset, 8),
        "nonzero_sample_ratio": round(nonzero_ratio, 8),
        "waveform_nonempty": bool(frames and peak > 0),
    }


def _raw_output_document(
    *,
    entry: Any,
    role: str,
    trace: TranscriptionTrace,
    adapter: SpeechModelAdapter,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "task_id": "TASK-003C",
        "captured_at": _utc_now(),
        "utterance_id": entry.utterance_id,
        "role": role,
        "audio_sha256": entry.audio_sha256,
        "generation_output_type": adapter.generation_output_type,
        "raw_model_output": trace.raw_model_output,
    }


def _adapter_trace_document(
    *,
    entry: Any,
    role: str,
    trace: TranscriptionTrace,
    latency_ms: float,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "task_id": "TASK-003C",
        "captured_at": _utc_now(),
        "utterance_id": entry.utterance_id,
        "role": role,
        "audio_sha256": entry.audio_sha256,
        "latency_ms": round(latency_ms, 3),
        "processor_call": trace.processor_call,
        "processor_inputs": trace.processor_inputs,
        "model_generate_call": trace.model_generate_call,
        "decoded_with_special_tokens": trace.decoded_with_special_tokens,
        "special_decode_error": trace.special_decode_error,
        "decoded_transcript": trace.decoded_transcript,
        "batch_decoded_transcripts": trace.batch_decoded_transcripts,
        "batch_decode_error": trace.batch_decode_error,
        "adapter_transcript": trace.adapter_transcript,
        "adapter_transformation": trace.adapter_transformation,
        "decoded_equals_adapter": trace.decoded_transcript == trace.adapter_transcript,
        "extraction_error": trace.extraction_error,
    }


def _target_summary(
    *,
    entry: Any,
    role: str,
    trace: TranscriptionTrace,
    latency_ms: float,
) -> dict[str, Any]:
    sequence = trace.raw_model_output.get("fields", {}).get("sequences")
    return {
        "utterance_id": entry.utterance_id,
        "role": role,
        "latency_ms": round(latency_ms, 3),
        "raw_sequence": sequence,
        "decoded_with_special_tokens": trace.decoded_with_special_tokens,
        "special_decode_error": trace.special_decode_error,
        "decoded_transcript": trace.decoded_transcript,
        "batch_decoded_transcripts": trace.batch_decoded_transcripts,
        "batch_decode_error": trace.batch_decode_error,
        "adapter_transcript": trace.adapter_transcript,
        "adapter_output_empty": trace.adapter_transcript == "",
        "decoded_equals_adapter": trace.decoded_transcript == trace.adapter_transcript,
        "extraction_error": trace.extraction_error,
    }


def _trace_error_message(trace: TranscriptionTrace) -> str:
    if trace.extraction_error is None:
        return "Adapter transcript is missing without an extraction error"
    return (
        f"{trace.extraction_error['error_type']}: "
        f"{trace.extraction_error['message']}"
    )


def _assess_diagnostics(
    *,
    probes: list[dict[str, Any]],
    targets: list[dict[str, Any]],
) -> dict[str, Any]:
    if any(
        probe.get("hash_matches_manifest") is not True
        or probe.get("sample_rate_hz") != 16_000
        or probe.get("channels") != 1
        or probe.get("sample_width_bits") != 16
        or probe.get("compression") != "NONE"
        or probe.get("sample_count", 0) <= 0
        for probe in probes
    ):
        return {
            "classification": "AUDIO_CONTRACT_ERROR",
            "evidence": ["At least one fixed target failed the recorded audio contract."],
            "scope": "automatic_contract_classification",
        }
    if any(
        target.get("extraction_error") is not None
        or target.get("decoded_equals_adapter") is not True
        for target in targets
    ):
        return {
            "classification": "ADAPTER_EXTRACTION_ERROR",
            "evidence": [
                "At least one official processor decode failed or differed from "
                "the Adapter transcript."
            ],
            "scope": "automatic_contract_classification",
        }

    empty_targets = [
        target
        for target in targets
        if target.get("role") == "baseline_empty_output"
    ]
    controls = [
        target
        for target in targets
        if target.get("role") == "baseline_nonempty_control"
    ]
    if (
        len(empty_targets) == 3
        and len(controls) == 1
        and all(target.get("decoded_transcript") == "" for target in empty_targets)
        and all(target.get("adapter_transcript") == "" for target in empty_targets)
        and controls[0].get("adapter_transcript") not in (None, "")
    ):
        return {
            "classification": "MODEL_RETURNED_EMPTY",
            "evidence": [
                "All three prior empty-output targets decoded to an empty string "
                "before the identity Adapter transformation.",
                "The fixed control produced a non-empty transcript in the same run.",
                "The run recorded ko-KR and the Offline generate path without "
                "streaming or context conditioning.",
            ],
            "scope": (
                "The empty transcript is localized to the official model/processor "
                "output path; this does not identify the model-internal reason."
            ),
        }
    if (
        len(empty_targets) == 3
        and all(target.get("adapter_transcript") not in (None, "") for target in empty_targets)
    ):
        return {
            "classification": "NOT_REPRODUCED",
            "evidence": [
                "All three prior empty-output targets produced non-empty transcripts."
            ],
            "scope": "Compare the captured environment with the original TASK-003B run.",
        }
    return {
        "classification": "UNRESOLVED",
        "evidence": [
            "The four-target output pattern does not support one automatic cause."
        ],
        "scope": "Review raw_model_output.jsonl and adapter_trace.jsonl.",
    }


def _prepare_output_directory(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in OUTPUT_NAMES if (output_dir / name).exists()]
    if existing:
        raise ContractError(
            f"append-only diagnostic files already exist: {existing}; "
            "choose a new output directory"
        )


def _create_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"task003c-{path.parent.name}-{id(path)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s")
    formatter.converter = time.gmtime
    handler = logging.FileHandler(path, mode="x", encoding="utf-8")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def _write_jsonl(path: Path, documents: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for document in documents:
            handle.write(
                json.dumps(document, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())


def _write_new_json(path: Path, document: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _sanitize_error_message(message: str) -> str:
    return message.replace(str(Path.home()), "<home>")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ContractError(f"{key} must be an object")
    return value


def _string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{key} must be a non-empty string")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
