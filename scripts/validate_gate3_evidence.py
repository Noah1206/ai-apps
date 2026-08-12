"""Validate a frozen Gate 3 criteria/evidence/assessment artifact set."""

from __future__ import annotations

import argparse
from pathlib import Path

from busan_lab.schemas.streaming import Gate3Assessment, Gate3Criteria, Gate3Evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--criteria", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--assessment", type=Path, required=True)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    Gate3Criteria.model_validate_json(arguments.criteria.read_text(encoding="utf-8"))
    Gate3Evidence.model_validate_json(arguments.evidence.read_text(encoding="utf-8"))
    Gate3Assessment.model_validate_json(arguments.assessment.read_text(encoding="utf-8"))
    print("Gate 3 criteria, evidence, and assessment are schema-valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
