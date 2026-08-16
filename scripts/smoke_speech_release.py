#!/usr/bin/env python3
"""Load the frozen offline ASR artifact and transcribe one non-test audio file."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

from busan_lab.speech_runtime import (
    DEFAULT_ADAPTER_SHA256,
    DEFAULT_MODEL_PATH,
    DEFAULT_MODEL_SHA256,
    DEFAULT_MODEL_VERSION,
    NemoOfflineASRBackend,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--expected-model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--expected-adapter-sha256", default=DEFAULT_ADAPTER_SHA256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.audio.is_file():
        parser.error(f"audio file does not exist: {args.audio}")
    if args.output and args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")

    backend = NemoOfflineASRBackend(
        model_path=args.model,
        expected_model_sha256=args.expected_model_sha256,
        expected_adapter_sha256=args.expected_adapter_sha256,
        model_version=args.model_version,
        device=args.device,
    )
    started_at = datetime.now(UTC)
    backend.load()
    loaded_at = datetime.now(UTC)
    inference = backend.transcribe(args.audio)
    completed_at = datetime.now(UTC)
    payload = {
        "status": "passed",
        "mode": "raw_offline_non_test_smoke",
        "started_at": started_at.isoformat().replace("+00:00", "Z"),
        "completed_at": completed_at.isoformat().replace("+00:00", "Z"),
        "python_version": platform.python_version(),
        "model_version": backend.model_version,
        "model_path": str(args.model),
        "model_sha256": args.expected_model_sha256,
        "adapter_sha256": args.expected_adapter_sha256,
        "audio_path": str(args.audio),
        "audio_sha256": sha256_file(args.audio),
        "transcript": inference.transcript,
        "model_load_ms": round((loaded_at - started_at).total_seconds() * 1000, 3),
        "latency_ms": round(inference.latency_ms, 3),
        "total_ms": round((completed_at - started_at).total_seconds() * 1000, 3),
        "device": args.device,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
