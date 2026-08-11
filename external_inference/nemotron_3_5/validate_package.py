#!/usr/bin/env python3
"""Validate an Audio Lab frozen benchmark export without loading a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from contract import ContractError
from run_inference import benchmark_summary, load_config, validate_from_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-package", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.example.yaml",
    )
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        benchmark = validate_from_config(args.benchmark_package, config)
    except (ContractError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps(benchmark_summary(benchmark), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
