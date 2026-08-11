#!/usr/bin/env python3
"""Validate predictions.jsonl against Audio Lab JSON Schema and benchmark coverage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from contract import (
    ContractError,
    load_jsonl,
    validate_nemotron_prediction_metadata,
    validate_prediction_documents,
)
from run_inference import load_config, validate_from_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--benchmark-package", required=True, type=Path)
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(__file__).resolve().parent / "schemas" / "predictions.schema.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.example.yaml",
    )
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        benchmark = validate_from_config(args.benchmark_package, config)
        records = validate_prediction_documents(
            load_jsonl(args.predictions),
            schema_path=args.schema,
            benchmark=benchmark,
        )
        model = config["model"]
        validate_nemotron_prediction_metadata(
            records,
            model_id=model["id"],
            resolved_revision=model["revision"],
            target_language=model["target_language"],
        )
    except (ContractError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "valid",
                "predictions": len(records),
                "benchmark_id": benchmark.benchmark_id,
                "benchmark_version": benchmark.benchmark_version,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
