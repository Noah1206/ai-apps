#!/usr/bin/env python3
"""Run 50 sequential raw ASR inferences on non-test Train audio."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from busan_lab.speech_runtime import (
    DEFAULT_ADAPTER_SHA256,
    DEFAULT_MODEL_PATH,
    DEFAULT_MODEL_SHA256,
    DEFAULT_MODEL_VERSION,
    NemoOfflineASRBackend,
)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/lab/gate2/final-attempt-v2/train/manifest.absolute.jsonl"),
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path("data/lab/gate2/final-attempt-v2/train/audio"),
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")

    selected: list[dict[str, object]] = []
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        duration = float(entry["duration"])
        if entry.get("split") != "train" or not 3 <= duration <= 8:
            continue
        audio_path = args.audio_root / Path(str(entry["audio_filepath"])).name
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        selected.append(
            {
                "utterance_id": str(entry["utterance_id"]),
                "audio_path": audio_path,
                "duration_ms": duration * 1000,
            }
        )
        if len(selected) == args.count:
            break
    if len(selected) != args.count:
        raise RuntimeError(f"found only {len(selected)} eligible Train files")

    backend = NemoOfflineASRBackend(
        model_path=args.model,
        expected_model_sha256=DEFAULT_MODEL_SHA256,
        expected_adapter_sha256=DEFAULT_ADAPTER_SHA256,
        model_version=DEFAULT_MODEL_VERSION,
    )
    started_at = datetime.now(UTC)
    load_started = time.perf_counter()
    backend.load()
    model_load_ms = (time.perf_counter() - load_started) * 1000
    torch.cuda.reset_peak_memory_stats()
    records: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for entry in selected:
        try:
            inference = backend.transcribe(Path(entry["audio_path"]))
            duration_ms = float(entry["duration_ms"])
            records.append(
                {
                    "utterance_id": entry["utterance_id"],
                    "duration_ms": round(duration_ms, 3),
                    "latency_ms": round(inference.latency_ms, 3),
                    "rtf": round(inference.latency_ms / duration_ms, 6),
                    "empty_output": not bool(inference.transcript.strip()),
                    "transcript": inference.transcript,
                }
            )
        except Exception as error:
            failures.append(
                {
                    "utterance_id": str(entry["utterance_id"]),
                    "error_type": type(error).__name__,
                    "message": str(error)[:500],
                }
            )
    completed_at = datetime.now(UTC)
    latencies = [float(record["latency_ms"]) for record in records]
    rtfs = [float(record["rtf"]) for record in records]
    empty_count = sum(bool(record["empty_output"]) for record in records)
    payload = {
        "status": (
            "passed"
            if len(records) == args.count and not failures and empty_count == 0
            else "failed"
        ),
        "mode": "raw_offline_non_test_train_stability",
        "started_at": started_at.isoformat().replace("+00:00", "Z"),
        "completed_at": completed_at.isoformat().replace("+00:00", "Z"),
        "model_version": backend.model_version,
        "model_sha256": DEFAULT_MODEL_SHA256,
        "requested_utterances": args.count,
        "successful_utterances": len(records),
        "failed_utterances": len(failures),
        "empty_outputs": empty_count,
        "model_load_ms": round(model_load_ms, 3),
        "latency_mean_ms": round(statistics.fmean(latencies), 3) if latencies else 0,
        "latency_p50_ms": round(statistics.median(latencies), 3) if latencies else 0,
        "latency_p95_ms": round(percentile(latencies, 0.95), 3) if latencies else 0,
        "rtf_mean": round(statistics.fmean(rtfs), 6) if rtfs else 0,
        "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "failures": failures,
        "records": records,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
