from pathlib import Path

from busan_lab.evaluation.calibration import calibrate_case, render_calibration_markdown
from busan_lab.schemas.calibration import (
    CalibratedForm,
    DialectCalibrationRule,
    DialectMatchKind,
    EvaluationCalibrationProfile,
    EvaluationCalibrationReport,
    ObservedError,
    SuspectedCause,
)
from busan_lab.schemas.common import ReviewStatus
from busan_lab.schemas.utterance import DialectExpressionLabel

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def profile(*rules: DialectCalibrationRule) -> EvaluationCalibrationProfile:
    return EvaluationCalibrationProfile(
        revision_id="test-calibration-v1",
        benchmark_id="fixture",
        benchmark_version="1.0.0",
        rules=rules,
    )


def label(
    surface: str,
    *,
    normalized_forms: tuple[str, ...] = (),
) -> DialectExpressionLabel:
    return DialectExpressionLabel(
        surface_form=surface,
        normalized_forms=normalized_forms,
    )


def error_types(
    reference: str, hypothesis: str, *labels: DialectExpressionLabel
) -> set[ObservedError]:
    _, observations, _ = calibrate_case(
        reference_surface_text=reference,
        hypothesis_surface_text=hypothesis,
        dialect_labels=labels,
        profile=profile(),
    )
    return {observation.observed_error for observation in observations}


def test_exact_dialect_surface_is_preserved() -> None:
    results, observations, candidates = calibrate_case(
        reference_surface_text="밥 묵었나?",
        hypothesis_surface_text="밥 묵었나",
        dialect_labels=(label("묵었나"),),
        profile=profile(),
    )

    assert results[0].match_kind is DialectMatchKind.EXACT_SURFACE
    assert {item.observed_error for item in observations} == {ObservedError.NO_ERROR}
    assert candidates == ()


def test_complete_dialect_omission_is_observed_without_inventing_a_cause() -> None:
    results, observations, candidates = calibrate_case(
        reference_surface_text="오늘 와 이리 춥노",
        hypothesis_surface_text="오늘 춥노",
        dialect_labels=(label("이리"),),
        profile=profile(),
    )

    assert results[0].match_kind is DialectMatchKind.MISSING
    lost = next(
        item
        for item in observations
        if item.observed_error is ObservedError.DIALECT_EXPRESSION_LOST
    )
    assert lost.suspected_cause is SuspectedCause.UNKNOWN
    assert ObservedError.WORD_OR_SYLLABLE_OMISSION in {item.observed_error for item in observations}
    assert candidates == ("MODEL",)


def test_partial_syllable_loss_and_phonetic_substitution_are_observable() -> None:
    errors = error_types(
        "국밥 하나 주이소",
        "국밥 하나 주위소",
        label("주이소"),
    )

    assert ObservedError.DIALECT_EXPRESSION_LOST in errors
    assert ObservedError.PHONETIC_SUBSTITUTION in errors
    assert ObservedError.ENDING_SUBSTITUTION in errors


def test_standard_form_is_observed_separately_from_suspected_model_bias() -> None:
    results, observations, candidates = calibrate_case(
        reference_surface_text="국밥 하나 주이소",
        hypothesis_surface_text="국밥 하나 주세요",
        dialect_labels=(label("주이소", normalized_forms=("주세요",)),),
        profile=profile(),
    )

    assert results[0].match_kind is DialectMatchKind.STANDARD_EQUIVALENT
    dialect_change = next(
        item for item in observations if item.observed_error is ObservedError.DIALECT_TO_STANDARD
    )
    assert dialect_change.suspected_cause is SuspectedCause.STANDARD_KOREAN_MODEL_BIAS
    assert "LANGUAGE_MODEL_BIAS" in candidates


def test_acoustic_like_substitution_does_not_become_language_model_bias() -> None:
    _, observations, candidates = calibrate_case(
        reference_surface_text="와따 맛있노",
        hypothesis_surface_text="다 마있노",
        dialect_labels=(label("와따"), label("맛있노")),
        profile=profile(),
    )

    assert ObservedError.PHONETIC_SUBSTITUTION in {item.observed_error for item in observations}
    assert "LANGUAGE_MODEL_BIAS" not in candidates
    assert candidates == ("MODEL",)


def test_spacing_only_difference_is_a_word_boundary_error() -> None:
    _, observations, candidates = calibrate_case(
        reference_surface_text="내일 같이 가자",
        hypothesis_surface_text="내일같이 가자",
        dialect_labels=(),
        profile=profile(),
    )

    assert {item.observed_error for item in observations} == {ObservedError.WORD_BOUNDARY_ERROR}
    assert candidates == ("DECODING",)


def test_approved_dialect_variant_is_accepted() -> None:
    calibration = profile(
        DialectCalibrationRule(
            surface_form="어데고",
            acceptable_variants=(CalibratedForm(form="어디고", status=ReviewStatus.APPROVED),),
        )
    )
    results, observations, candidates = calibrate_case(
        reference_surface_text="니 지금 어데고",
        hypothesis_surface_text="니 지금 어디고",
        dialect_labels=(label("어데고"),),
        profile=calibration,
    )

    assert results[0].match_kind is DialectMatchKind.ACCEPTABLE_VARIANT
    assert results[0].mapping_status is ReviewStatus.APPROVED
    assert ObservedError.AMBIGUOUS_VARIANT not in {item.observed_error for item in observations}
    assert candidates == ()


def test_candidate_dialect_variant_is_uncertain_not_a_forced_failure() -> None:
    calibration = profile(
        DialectCalibrationRule(
            surface_form="어데고",
            acceptable_variants=(CalibratedForm(form="어디고"),),
        )
    )
    results, observations, candidates = calibrate_case(
        reference_surface_text="니 지금 어데고",
        hypothesis_surface_text="니 지금 어디고",
        dialect_labels=(label("어데고"),),
        profile=calibration,
    )

    assert results[0].match_kind is DialectMatchKind.ACCEPTABLE_VARIANT
    assert results[0].mapping_status is ReviewStatus.CANDIDATE
    assert ObservedError.AMBIGUOUS_VARIANT in {item.observed_error for item in observations}
    assert candidates == ()


def test_first_word_omission_is_observed() -> None:
    errors = error_types("지금 뭐 하노", "뭐 하노", label("하노"))

    assert ObservedError.WORD_OR_SYLLABLE_OMISSION in errors


def test_ending_syllable_omission_is_observed() -> None:
    errors = error_types("내일 같이 가재이", "내일 같이 가 제", label("가재이"))

    assert ObservedError.DIALECT_EXPRESSION_LOST in errors
    assert ObservedError.WORD_OR_SYLLABLE_OMISSION in errors
    assert ObservedError.ENDING_SUBSTITUTION in errors


def test_reviewed_standard_candidates_cover_task_002_regressions() -> None:
    calibration = profile(
        DialectCalibrationRule(
            surface_form="아이가",
            standard_equivalents=(CalibratedForm(form="이거"),),
        ),
        DialectCalibrationRule(
            surface_form="앉으이소",
            standard_equivalents=(CalibratedForm(form="앉아있어"),),
        ),
    )

    for reference, hypothesis, surface in (
        ("마 괜찮다 아이가", "아 괜찮다 이거", "아이가"),
        ("여기 좀 앉으이소", "아이 여기 좀 앉아있어", "앉으이소"),
    ):
        results, _, candidates = calibrate_case(
            reference_surface_text=reference,
            hypothesis_surface_text=hypothesis,
            dialect_labels=(label(surface),),
            profile=calibration,
        )
        assert results[0].match_kind is DialectMatchKind.STANDARD_EQUIVALENT
        assert "LANGUAGE_MODEL_BIAS" in candidates


def test_task_002_calibration_revision_is_preserved_as_a_separate_report() -> None:
    report_path = (
        PROJECT_ROOT
        / "data/lab/reports"
        / (
            "task-002-nvidia-korean-conformer-ctc-pretrained-v0--"
            "task-003a-surface-asr-evaluation-v1.json"
        )
    )
    report = EvaluationCalibrationReport.model_validate_json(
        report_path.read_text(encoding="utf-8")
    )

    assert report.evaluation_revision == "task-003a-surface-asr-evaluation-v1"
    assert report.source_report_id == "report-54bac50f-36da-4f93-ba6b-efe249dc62d8"
    assert report.legacy_automatic_metrics.context_overcorrection_rate == 0
    assert report.calibrated_metrics.standard_equivalent_candidate_count == 2
    assert report.calibrated_metrics.ambiguous_expression_count == 1
    assert report.human_comparison.automatic_human_match_count == 7
    assert report.human_comparison.uncertain_count == 1
    assert len(report.cases) == 10
    assert report.task_003b_prediction_contract_compatible is True

    neutral_markdown = render_calibration_markdown(report)
    assert "Whisper" not in neutral_markdown
    assert "최신 계획 문서의 승인된 TASK" in neutral_markdown

    nemotron_markdown = render_calibration_markdown(
        report,
        next_experiment="TASK-003B Nemotron pretrained baseline",
    )
    assert "TASK-003B Nemotron pretrained baseline" in nemotron_markdown
