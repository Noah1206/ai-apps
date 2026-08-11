"""Surface ASR evaluation engine."""

from busan_lab.evaluation.metrics import (
    aggregate_cases,
    character_error_rate,
    evaluate_case,
    evaluate_dialect_preservation,
)
from busan_lab.evaluation.runner import BenchmarkRunner

__all__ = [
    "BenchmarkRunner",
    "aggregate_cases",
    "character_error_rate",
    "evaluate_case",
    "evaluate_dialect_preservation",
]
