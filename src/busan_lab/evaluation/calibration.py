"""Re-evaluate stored predictions without changing inference or review evidence."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

from busan_lab.evaluation.metrics import character_error_rate, normalize_for_cer
from busan_lab.schemas.calibration import (
    CalibratedAggregateMetrics,
    CalibratedDialectResult,
    CalibratedEvaluationCase,
    CalibratedForm,
    DialectCalibrationRule,
    DialectMatchKind,
    ErrorObservation,
    EvaluationCalibrationProfile,
    EvaluationCalibrationReport,
    HumanComparisonOutcome,
    HumanComparisonSummary,
    ObservedError,
    PredictionDiagnostics,
    SuspectedCause,
)
from busan_lab.schemas.common import ReviewStatus
from busan_lab.schemas.evaluation import BaselineReport, HumanReviewedBaselineReport
from busan_lab.schemas.experiment import HumanReview, ReviewVerdict, StoredPrediction
from busan_lab.schemas.utterance import DialectExpressionLabel
from busan_lab.storage import LabStorage

_NON_ACTIONABLE_ERRORS = frozenset(
    {
        ObservedError.NO_ERROR,
        ObservedError.AMBIGUOUS_VARIANT,
    }
)


def load_calibration_profile(path: Path) -> EvaluationCalibrationProfile:
    return EvaluationCalibrationProfile.model_validate_json(
        path.expanduser().read_text(encoding="utf-8")
    )


def diagnose_prediction(
    storage: LabStorage,
    prediction: StoredPrediction,
) -> PredictionDiagnostics:
    """Calibrate one stored prediction without changing its persisted evidence."""

    experiment = storage.load_experiment(prediction.experiment_id)
    matching_profiles = tuple(
        profile
        for profile in (
            load_calibration_profile(path)
            for path in sorted((storage.root / "calibrations").glob("*.json"))
        )
        if experiment.benchmark_id is not None
        and experiment.benchmark_version is not None
        and profile.benchmark_id == experiment.benchmark_id
        and profile.benchmark_version == experiment.benchmark_version
    )
    matched_profile = (
        max(
            matching_profiles,
            key=lambda profile: (profile.created_at, profile.revision_id),
        )
        if matching_profiles
        else None
    )
    profile = matched_profile or EvaluationCalibrationProfile(
        revision_id="live-observation-only",
        benchmark_id=experiment.benchmark_id or "unversioned",
        benchmark_version=experiment.benchmark_version or "unversioned",
        rules=(),
        notes=("저장된 Prediction을 변경하지 않는 UI 관찰 전용 profile",),
    )
    dialect_labels = tuple(
        DialectExpressionLabel(
            surface_form=result.surface_form,
            normalized_forms=result.normalized_forms,
            status=result.label_status,
        )
        for result in prediction.evaluation.dialect.results
    )
    _, observations, automatic_candidates = calibrate_case(
        reference_surface_text=prediction.evaluation.reference_surface_text,
        hypothesis_surface_text=prediction.evaluation.hypothesis_surface_text,
        dialect_labels=dialect_labels,
        profile=profile,
    )
    return PredictionDiagnostics(
        prediction_id=prediction.prediction_id,
        calibration_revision=(
            matched_profile.revision_id if matched_profile is not None else None
        ),
        observations=observations,
        automatic_failure_candidates=automatic_candidates,
    )


def calibrate_case(
    *,
    reference_surface_text: str,
    hypothesis_surface_text: str,
    dialect_labels: Sequence[DialectExpressionLabel],
    profile: EvaluationCalibrationProfile,
) -> tuple[
    tuple[CalibratedDialectResult, ...],
    tuple[ErrorObservation, ...],
    tuple[str, ...],
]:
    """Return observable errors and explicitly tentative causes for one output."""

    rules = _rules_by_surface(profile.rules)
    expression_results = tuple(
        _calibrate_expression(hypothesis_surface_text, label, rules.get(label.surface_form))
        for label in dialect_labels
    )
    observations = [
        observation for result in expression_results for observation in result.observations
    ]
    comparison_hypothesis = hypothesis_surface_text
    for result in expression_results:
        if (
            result.match_kind is DialectMatchKind.ACCEPTABLE_VARIANT
            and result.matched_form is not None
            and result.matched_form in comparison_hypothesis
        ):
            comparison_hypothesis = comparison_hypothesis.replace(
                result.matched_form,
                result.surface_form,
                1,
            )
    observations.extend(_case_edit_observations(reference_surface_text, comparison_hypothesis))
    deduplicated = _deduplicate_observations(observations)
    return (
        expression_results,
        deduplicated,
        _automatic_failure_candidates(deduplicated),
    )


def recalibrate_experiment(
    storage: LabStorage,
    experiment_id: str,
    profile: EvaluationCalibrationProfile,
    *,
    next_experiment: str | None = None,
) -> tuple[EvaluationCalibrationReport, Path, Path]:
    """Create a new evaluation revision from immutable stored evidence."""

    source = _load_source_report(storage, experiment_id)
    experiment = storage.load_experiment(experiment_id)
    manifest = storage.load_manifest(source.benchmark_id, source.benchmark_version)
    if (
        profile.benchmark_id != manifest.benchmark_id
        or profile.benchmark_version != manifest.benchmark_version
    ):
        raise ValueError("calibration profile does not match the source benchmark")
    if experiment.model != source.model:
        raise ValueError("source report model does not match the experiment")

    entries = {entry.utterance_id: entry for entry in manifest.entries}
    predictions = tuple(
        prediction
        for prediction in storage.list_predictions()
        if prediction.experiment_id == experiment_id
    )
    predictions_by_utterance = _unique_predictions(predictions)
    source_cases = {case.utterance_id: case for case in source.cases}
    expected_ids = set(entries)
    if set(predictions_by_utterance) != expected_ids or set(source_cases) != expected_ids:
        raise ValueError(
            "benchmark, source report, and predictions do not cover the same utterances"
        )

    reviews_by_prediction: dict[UUID, list[HumanReview]] = defaultdict(list)
    for review in storage.list_reviews():
        reviews_by_prediction[review.prediction_id].append(review)

    calibrated_cases: list[CalibratedEvaluationCase] = []
    latest_reviews: list[HumanReview] = []
    for source_case in source.cases:
        entry = entries[source_case.utterance_id]
        prediction = predictions_by_utterance[source_case.utterance_id]
        if prediction.evaluation != source_case:
            raise ValueError("stored prediction does not match the source report")
        revisions = reviews_by_prediction[prediction.prediction_id]
        if not revisions:
            raise ValueError(f"missing human review for utterance {entry.utterance_id}")
        latest = max(revisions, key=lambda review: (review.created_at, str(review.review_id)))
        _validate_review_identity(latest, prediction)
        latest_reviews.append(latest)

        dialect_results, observations, automatic_candidates = calibrate_case(
            reference_surface_text=source_case.reference_surface_text,
            hypothesis_surface_text=source_case.hypothesis_surface_text,
            dialect_labels=entry.dialect_expressions,
            profile=profile,
        )
        calibrated_cases.append(
            CalibratedEvaluationCase(
                prediction_id=prediction.prediction_id,
                utterance_id=source_case.utterance_id,
                reference_surface_text=source_case.reference_surface_text,
                hypothesis_surface_text=source_case.hypothesis_surface_text,
                legacy_dialect_results=source_case.dialect.results,
                legacy_automatic_failure_candidates=prediction.automatic_failure_candidates,
                calibrated_dialect_results=dialect_results,
                observations=observations,
                automatic_failure_candidates=automatic_candidates,
                latest_review_id=latest.review_id,
                review_revision_count=len(revisions),
                human_verdict=latest.verdict.value,
                human_confirmed_failure_types=latest.confirmed_failure_types,
                human_review_notes=latest.notes,
                comparison_outcome=_compare_with_human(automatic_candidates, latest),
            )
        )

    cases = tuple(calibrated_cases)
    report = EvaluationCalibrationReport(
        report_id=f"{source.report_id}--{profile.revision_id}",
        evaluation_revision=profile.revision_id,
        source_report_id=source.report_id,
        source_reviewed_report_id=_reviewed_report_id(storage, experiment_id, source.report_id),
        experiment_id=experiment_id,
        benchmark_id=source.benchmark_id,
        benchmark_version=source.benchmark_version,
        created_at=max(
            source.created_at,
            profile.created_at,
            *(review.created_at for review in latest_reviews),
        ),
        model=source.model,
        legacy_automatic_metrics=source.metrics,
        calibrated_metrics=_aggregate_calibrated(cases),
        human_comparison=_aggregate_human_comparison(cases),
        cases=cases,
        limitations=(
            "후보 변이와 표준어 대응은 Pilot 검수 근거이며 언어학적으로 승인된 "
            "사전으로 해석하지 않는다.",
            "텍스트 차이만으로 음향·디코더·언어모델 원인을 확정할 수 없어 "
            "suspected_cause는 후보로만 사용한다.",
            "10개 Pilot 결과이므로 모델 성능에 대한 통계적 일반화를 하지 않는다.",
            "candidate 상태의 허용 변이는 실패와 보존 양쪽에서 제외한다.",
        ),
        task_003b_prediction_contract_compatible=True,
    )
    safe_experiment = _safe_name(experiment_id)
    safe_revision = _safe_name(profile.revision_id)
    stem = f"{safe_experiment}--{safe_revision}"
    json_path = storage.reports_dir / f"{stem}.json"
    markdown_path = storage.reports_dir / f"{stem}.md"
    _write(json_path, report.model_dump_json(indent=2) + "\n")
    _write(
        markdown_path,
        render_calibration_markdown(report, next_experiment=next_experiment),
    )
    return report, json_path, markdown_path


def render_calibration_markdown(
    report: EvaluationCalibrationReport,
    *,
    next_experiment: str | None = None,
) -> str:
    legacy = report.legacy_automatic_metrics
    calibrated = report.calibrated_metrics
    comparison = report.human_comparison
    lines = [
        "# TASK-003A Surface ASR Evaluation Calibration",
        "",
        "## 재평가 조건",
        "",
        f"- Experiment: `{report.experiment_id}`",
        f"- Benchmark: `{report.benchmark_id}@{report.benchmark_version}`",
        f"- Evaluation revision: `{report.evaluation_revision}`",
        f"- Source report: `{report.source_report_id}`",
        f"- Source reviewed report: `{report.source_reviewed_report_id or '미기록'}`",
        "- 재추론: 하지 않음",
        "- 기존 Prediction / Human Review 수정: 하지 않음",
        "",
        "## 기존 자동 평가 vs 보정 평가",
        "",
        "| 지표 | 기존 | 보정 |",
        "|---|---:|---:|",
        f"| CER | {legacy.cer:.4f} | {legacy.cer:.4f} |",
        f"| 방언 보존율 | {legacy.dialect_preservation_rate:.4f} | "
        f"{calibrated.dialect_preservation_rate:.4f} |",
        f"| Context overcorrection | {legacy.context_overcorrection_rate:.4f} | "
        f"{calibrated.context_overcorrection_candidate_rate:.4f} (candidate) |",
        f"| 평가 표현 수 | {calibrated.reference_expression_count} | "
        f"{calibrated.evaluated_expression_count} |",
        f"| 불확실 변이 | 0 | {calibrated.ambiguous_expression_count} |",
        "",
        "보정 과보정 값은 확정률이 아니라 표준어 방향 원인 **후보** 비율이다.",
        "",
        "## 자동 후보와 최신 사람 검수 비교",
        "",
        f"- 일치: {comparison.automatic_human_match_count}",
        f"- 불일치: {comparison.automatic_human_mismatch_count}",
        f"- 자동 미탐지: {comparison.automatic_missed_count}",
        f"- 자동 과탐지: {comparison.automatic_false_positive_count}",
        f"- 정상 일치: {comparison.no_error_agreement_count}",
        f"- 불확실: {comparison.uncertain_count}",
        "",
        "## 발화별 차이",
        "",
        "| Surface 정답 | 모델 예측 | 기존 자동 | 보정 관찰 오류 | 원인 후보 | 사람 검수 | 비교 |",
        "|---|---|---|---|---|---|---|",
    ]
    for case in report.cases:
        legacy_status = (
            ", ".join(
                f"{result.surface_form}:{result.match_status.value}"
                for result in case.legacy_dialect_results
            )
            or "표현 없음"
        )
        observed = (
            ", ".join(
                _observation_label(observation)
                for observation in case.observations
                if observation.observed_error is not ObservedError.NO_ERROR
            )
            or "NO_ERROR"
        )
        causes = (
            ", ".join(
                dict.fromkeys(
                    observation.suspected_cause.value
                    for observation in case.observations
                    if observation.suspected_cause is not SuspectedCause.UNKNOWN
                )
            )
            or "UNKNOWN/없음"
        )
        human = str(case.human_verdict)
        if case.human_confirmed_failure_types:
            human += ":" + ",".join(case.human_confirmed_failure_types)
        lines.append(
            "| "
            + " | ".join(
                (
                    _cell(case.reference_surface_text),
                    _cell(case.hypothesis_surface_text),
                    _cell(legacy_status),
                    _cell(observed),
                    _cell(causes),
                    _cell(human),
                    case.comparison_outcome.value,
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 알려진 한계",
            "",
            *(f"- {item}" for item in report.limitations),
            "",
            "## TASK-003B 준비 상태",
            "",
            "- 기존 Prediction Import와 모델 메타데이터 계약은 특정 ASR 구조에 "
            "종속되지 않아 두 번째 모델에도 재사용할 수 있다.",
            (
                f"- 다음 실험: {next_experiment}"
                if next_experiment
                else "- 다음 실험은 최신 계획 문서의 승인된 TASK를 따른다."
            ),
            "- 이 보고서는 모델 선택이나 Fine-tuning 결정을 확정하지 않는다.",
            "",
        ]
    )
    return "\n".join(lines)


def _calibrate_expression(
    hypothesis: str,
    label: DialectExpressionLabel,
    rule: DialectCalibrationRule | None,
) -> CalibratedDialectResult:
    if _contains_form(hypothesis, label.surface_form):
        return CalibratedDialectResult(
            surface_form=label.surface_form,
            match_kind=DialectMatchKind.EXACT_SURFACE,
            matched_form=label.surface_form,
            mapping_status=label.status,
            observations=(
                ErrorObservation(
                    observed_error=ObservedError.NO_ERROR,
                    surface_form=label.surface_form,
                    matched_form=label.surface_form,
                    mapping_status=label.status,
                ),
            ),
        )

    acceptable_variants = rule.acceptable_variants if rule is not None else ()
    acceptable_match = _first_matching_form(hypothesis, acceptable_variants)
    if acceptable_match is not None:
        if acceptable_match.status is ReviewStatus.APPROVED:
            observed_error = ObservedError.NO_ERROR
        else:
            observed_error = ObservedError.AMBIGUOUS_VARIANT
        return CalibratedDialectResult(
            surface_form=label.surface_form,
            match_kind=DialectMatchKind.ACCEPTABLE_VARIANT,
            matched_form=acceptable_match.form,
            mapping_status=acceptable_match.status,
            observations=(
                ErrorObservation(
                    observed_error=observed_error,
                    surface_form=label.surface_form,
                    matched_form=acceptable_match.form,
                    mapping_status=acceptable_match.status,
                    evidence=acceptable_match.notes,
                ),
            ),
        )

    standard_forms = tuple(
        CalibratedForm(
            form=form,
            status=label.status,
            notes="Benchmark normalized_forms",
        )
        for form in label.normalized_forms
    )
    if rule is not None:
        standard_forms += rule.standard_equivalents
    standard_match = _first_matching_form(hypothesis, standard_forms)
    if standard_match is not None:
        return CalibratedDialectResult(
            surface_form=label.surface_form,
            match_kind=DialectMatchKind.STANDARD_EQUIVALENT,
            matched_form=standard_match.form,
            mapping_status=standard_match.status,
            observations=(
                ErrorObservation(
                    observed_error=ObservedError.DIALECT_TO_STANDARD,
                    suspected_cause=SuspectedCause.STANDARD_KOREAN_MODEL_BIAS,
                    surface_form=label.surface_form,
                    matched_form=standard_match.form,
                    mapping_status=standard_match.status,
                    evidence=standard_match.notes,
                ),
            ),
        )

    return CalibratedDialectResult(
        surface_form=label.surface_form,
        match_kind=DialectMatchKind.MISSING,
        observations=(
            ErrorObservation(
                observed_error=ObservedError.DIALECT_EXPRESSION_LOST,
                surface_form=label.surface_form,
            ),
        ),
    )


def _case_edit_observations(reference: str, hypothesis: str) -> tuple[ErrorObservation, ...]:
    normalized_reference = normalize_for_cer(reference)
    normalized_hypothesis = normalize_for_cer(hypothesis)
    _, edits = character_error_rate(reference, hypothesis)
    observations: list[ErrorObservation] = []
    if normalized_reference == normalized_hypothesis and _tokens_without_punctuation(
        reference
    ) != _tokens_without_punctuation(hypothesis):
        observations.append(
            ErrorObservation(
                observed_error=ObservedError.WORD_BOUNDARY_ERROR,
                evidence="문자열은 같지만 공백 경계가 다름",
            )
        )
    if edits.substitutions:
        observations.append(
            ErrorObservation(
                observed_error=ObservedError.PHONETIC_SUBSTITUTION,
                evidence=f"문자 치환 {edits.substitutions}개",
            )
        )
    if edits.deletions:
        observations.append(
            ErrorObservation(
                observed_error=ObservedError.WORD_OR_SYLLABLE_OMISSION,
                evidence=f"문자 누락 {edits.deletions}개",
            )
        )
    if edits.insertions:
        observations.append(
            ErrorObservation(
                observed_error=ObservedError.WORD_OR_SYLLABLE_INSERTION,
                evidence=f"문자 삽입 {edits.insertions}개",
            )
        )
    ending_size = min(3, len(normalized_reference), len(normalized_hypothesis))
    if (
        ending_size
        and normalized_reference != normalized_hypothesis
        and normalized_reference[-ending_size:] != normalized_hypothesis[-ending_size:]
    ):
        observations.append(
            ErrorObservation(
                observed_error=ObservedError.ENDING_SUBSTITUTION,
                evidence="문장 끝 음절열이 정답과 다름",
            )
        )
    return tuple(observations)


def _automatic_failure_candidates(
    observations: Sequence[ErrorObservation],
) -> tuple[str, ...]:
    actionable = tuple(
        observation
        for observation in observations
        if observation.observed_error not in _NON_ACTIONABLE_ERRORS
    )
    if not actionable:
        return ()
    candidates: list[str] = []
    if any(
        observation.suspected_cause is SuspectedCause.STANDARD_KOREAN_MODEL_BIAS
        for observation in actionable
    ):
        candidates.append("LANGUAGE_MODEL_BIAS")
    non_boundary = any(
        observation.observed_error
        not in {
            ObservedError.WORD_BOUNDARY_ERROR,
            ObservedError.DIALECT_TO_STANDARD,
        }
        for observation in actionable
    )
    if non_boundary:
        candidates.append("MODEL")
    elif any(
        observation.observed_error is ObservedError.WORD_BOUNDARY_ERROR
        for observation in actionable
    ):
        candidates.append("DECODING")
    return tuple(candidates)


def _compare_with_human(
    automatic_candidates: Sequence[str],
    review: HumanReview,
) -> HumanComparisonOutcome:
    if review.verdict is ReviewVerdict.UNCERTAIN:
        return HumanComparisonOutcome.UNCERTAIN
    automatic = set(automatic_candidates)
    human = set(review.confirmed_failure_types)
    if review.verdict is ReviewVerdict.REJECTED:
        if automatic:
            return HumanComparisonOutcome.FALSE_POSITIVE
        return HumanComparisonOutcome.NO_ERROR_AGREEMENT
    if not automatic:
        return HumanComparisonOutcome.MISSED
    if automatic & human:
        return HumanComparisonOutcome.MATCH
    return HumanComparisonOutcome.MISMATCH


def _aggregate_calibrated(
    cases: Sequence[CalibratedEvaluationCase],
) -> CalibratedAggregateMetrics:
    results = tuple(result for case in cases for result in case.calibrated_dialect_results)
    ambiguous = sum(
        result.match_kind is DialectMatchKind.ACCEPTABLE_VARIANT
        and result.mapping_status is not ReviewStatus.APPROVED
        for result in results
    )
    exact = sum(result.match_kind is DialectMatchKind.EXACT_SURFACE for result in results)
    acceptable = sum(
        result.match_kind is DialectMatchKind.ACCEPTABLE_VARIANT
        and result.mapping_status is ReviewStatus.APPROVED
        for result in results
    )
    standard = sum(result.match_kind is DialectMatchKind.STANDARD_EQUIVALENT for result in results)
    missing = sum(result.match_kind is DialectMatchKind.MISSING for result in results)
    evaluated = len(results) - ambiguous
    return CalibratedAggregateMetrics(
        reference_expression_count=len(results),
        evaluated_expression_count=evaluated,
        ambiguous_expression_count=ambiguous,
        exact_preserved_count=exact,
        acceptable_variant_count=acceptable,
        standard_equivalent_candidate_count=standard,
        missing_expression_count=missing,
        dialect_preservation_rate=(exact + acceptable) / evaluated if evaluated else 1,
        context_overcorrection_candidate_rate=standard / evaluated if evaluated else 0,
    )


def _aggregate_human_comparison(
    cases: Sequence[CalibratedEvaluationCase],
) -> HumanComparisonSummary:
    counts = Counter(case.comparison_outcome for case in cases)
    return HumanComparisonSummary(
        automatic_human_match_count=counts[HumanComparisonOutcome.MATCH],
        automatic_human_mismatch_count=counts[HumanComparisonOutcome.MISMATCH],
        automatic_missed_count=counts[HumanComparisonOutcome.MISSED],
        automatic_false_positive_count=counts[HumanComparisonOutcome.FALSE_POSITIVE],
        no_error_agreement_count=counts[HumanComparisonOutcome.NO_ERROR_AGREEMENT],
        uncertain_count=counts[HumanComparisonOutcome.UNCERTAIN],
    )


def _rules_by_surface(
    rules: Sequence[DialectCalibrationRule],
) -> dict[str, DialectCalibrationRule]:
    by_surface: dict[str, DialectCalibrationRule] = {}
    for rule in rules:
        if rule.surface_form in by_surface:
            raise ValueError(f"duplicate calibration rule: {rule.surface_form}")
        by_surface[rule.surface_form] = rule
    return by_surface


def _first_matching_form(
    hypothesis: str,
    forms: Sequence[CalibratedForm],
) -> CalibratedForm | None:
    return next((form for form in forms if _contains_form(hypothesis, form.form)), None)


def _contains_form(hypothesis: str, form: str) -> bool:
    return normalize_for_cer(form) in normalize_for_cer(hypothesis)


def _tokens_without_punctuation(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFC", text)
    without_punctuation = "".join(
        character for character in normalized if unicodedata.category(character)[0] != "P"
    )
    return tuple(without_punctuation.split())


def _deduplicate_observations(
    observations: Sequence[ErrorObservation],
) -> tuple[ErrorObservation, ...]:
    deduplicated: list[ErrorObservation] = []
    seen: set[tuple[object, ...]] = set()
    for observation in observations:
        key = (
            observation.observed_error,
            observation.suspected_cause,
            observation.surface_form,
            observation.matched_form,
            observation.mapping_status,
            observation.evidence,
        )
        if key not in seen:
            seen.add(key)
            deduplicated.append(observation)
    return tuple(deduplicated)


def _unique_predictions(
    predictions: Sequence[StoredPrediction],
) -> dict[UUID, StoredPrediction]:
    by_utterance: dict[UUID, StoredPrediction] = {}
    for prediction in predictions:
        if prediction.utterance_id in by_utterance:
            raise ValueError("experiment contains duplicate predictions for an utterance")
        by_utterance[prediction.utterance_id] = prediction
    return by_utterance


def _validate_review_identity(review: HumanReview, prediction: StoredPrediction) -> None:
    if review.prediction_id != prediction.prediction_id:
        raise ValueError("review does not match its prediction")
    if review.utterance_id != prediction.utterance_id:
        raise ValueError("review utterance does not match its prediction")


def _load_source_report(storage: LabStorage, experiment_id: str) -> BaselineReport:
    reports = tuple(
        BaselineReport.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(storage.reports_dir.glob("report-*.json"))
    )
    matching = tuple(report for report in reports if report.experiment_id == experiment_id)
    if len(matching) != 1:
        raise ValueError(
            f"expected exactly one automatic baseline report for {experiment_id}, "
            f"found {len(matching)}"
        )
    return matching[0]


def _reviewed_report_id(
    storage: LabStorage,
    experiment_id: str,
    source_report_id: str,
) -> str | None:
    for path in sorted(storage.reports_dir.glob("*--reviewed.json")):
        report = HumanReviewedBaselineReport.model_validate_json(path.read_text(encoding="utf-8"))
        if report.experiment_id == experiment_id and report.source_report_id == source_report_id:
            return report.report_id
    return None


def _observation_label(observation: ErrorObservation) -> str:
    if observation.surface_form is None:
        return observation.observed_error.value
    return f"{observation.surface_form}:{observation.observed_error.value}"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-") or "evaluation"


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def _write(path: Path, content: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
