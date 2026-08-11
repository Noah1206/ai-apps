from uuid import uuid4

import pytest

from busan_lab.evaluation.metrics import (
    aggregate_cases,
    character_error_rate,
    evaluate_case,
    evaluate_dialect_preservation,
)
from busan_lab.schemas.asr import ModelDescriptor
from busan_lab.schemas.common import ReviewStatus
from busan_lab.schemas.evaluation import DialectMatchStatus
from busan_lab.schemas.utterance import DialectExpressionLabel


def test_cer_ignores_spacing_and_punctuation_but_not_surface_characters() -> None:
    cer, edits = character_error_rate("니 지금 어데고?", "니지금어디고")

    assert edits.substitutions == 1
    assert edits.deletions == 0
    assert edits.insertions == 0
    assert cer == pytest.approx(1 / 6)


def test_dialect_preservation_detects_standard_overcorrection() -> None:
    metric = evaluate_dialect_preservation(
        "국밥 하나 주세요",
        [
            DialectExpressionLabel(
                surface_form="주이소",
                normalized_forms=("주세요",),
                status=ReviewStatus.CANDIDATE,
            )
        ],
    )

    assert metric.preservation_rate == 0
    assert metric.overcorrection_rate == 1
    assert metric.approved_expression_count == 0
    assert metric.results[0].match_status is DialectMatchStatus.OVERCORRECTED


def test_high_confidence_wrong_is_separate_from_error_size() -> None:
    case = evaluate_case(
        utterance_id=uuid4(),
        reference_surface_text="국밥 하나 주이소",
        normalized_meaning="국밥 하나 주세요",
        hypothesis_surface_text="국밥 하나 주세요",
        confidence=0.91,
        dialect_labels=(
            DialectExpressionLabel(
                surface_form="주이소",
                normalized_forms=("주세요",),
            ),
        ),
        model=ModelDescriptor(name="fixture", version="v0"),
    )

    assert case.cer > 0
    assert case.high_confidence_wrong is True
    assert case.dialect.overcorrection_rate == 1


def test_empty_dialect_labels_are_neutral() -> None:
    metric = evaluate_dialect_preservation("아무 문장", [])

    assert metric.preservation_rate == 1
    assert metric.overcorrection_rate == 0


def test_aggregate_requires_one_latency_per_case() -> None:
    with pytest.raises(ValueError, match="one latency"):
        aggregate_cases([], [10])
