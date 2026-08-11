"""Finalize an automatic baseline with the latest human review evidence."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from uuid import UUID

from busan_lab.schemas.evaluation import (
    BaselineReport,
    FailureTypeCount,
    HumanReviewedBaselineReport,
    HumanReviewSummary,
    ReviewedEvaluationCase,
)
from busan_lab.schemas.experiment import HumanReview, ReviewVerdict, StoredPrediction
from busan_lab.storage import LabStorage

VALID_FAILURE_TYPES = frozenset(
    {
        "DATA",
        "LABEL",
        "AUDIO",
        "TOKENIZER",
        "MODEL",
        "DECODING",
        "LANGUAGE_MODEL_BIAS",
        "CALIBRATION",
    }
)


def finalize_human_reviewed_report(
    storage: LabStorage,
    experiment_id: str,
) -> tuple[HumanReviewedBaselineReport, Path, Path]:
    source = _load_source_report(storage, experiment_id)
    experiment = storage.load_experiment(experiment_id)
    if experiment.model != source.model:
        raise ValueError("baseline report model does not match the experiment")
    if (
        experiment.benchmark_id != source.benchmark_id
        or experiment.benchmark_version != source.benchmark_version
    ):
        raise ValueError("baseline report benchmark does not match the experiment")

    predictions = tuple(
        prediction
        for prediction in storage.list_predictions()
        if prediction.experiment_id == experiment_id
    )
    by_utterance = {prediction.utterance_id: prediction for prediction in predictions}
    if len(by_utterance) != len(predictions):
        raise ValueError("experiment contains duplicate predictions for an utterance")
    expected_utterances = {case.utterance_id for case in source.cases}
    if set(by_utterance) != expected_utterances:
        raise ValueError("experiment predictions do not match the baseline report cases")

    reviews_by_prediction: dict[UUID, list[HumanReview]] = defaultdict(list)
    for review in storage.list_reviews():
        reviews_by_prediction[review.prediction_id].append(review)

    reviewed_cases: list[ReviewedEvaluationCase] = []
    latest_reviews: list[HumanReview] = []
    failure_counts: Counter[str] = Counter()
    for case in source.cases:
        prediction = by_utterance[case.utterance_id]
        if prediction.evaluation != case or prediction.evaluation.model != source.model:
            raise ValueError("stored prediction does not match the baseline report")
        revisions = reviews_by_prediction[prediction.prediction_id]
        if not revisions:
            raise ValueError(f"missing human review for utterance {case.utterance_id}")
        latest = max(revisions, key=lambda review: (review.created_at, str(review.review_id)))
        _validate_review(latest, prediction)
        latest_reviews.append(latest)
        if latest.verdict is ReviewVerdict.CONFIRMED:
            failure_counts.update(latest.confirmed_failure_types)
        reviewed_cases.append(
            ReviewedEvaluationCase(
                prediction_id=prediction.prediction_id,
                utterance_id=case.utterance_id,
                reference_surface_text=case.reference_surface_text,
                hypothesis_surface_text=case.hypothesis_surface_text,
                cer=case.cer,
                dialect=case.dialect,
                automatic_failure_candidates=prediction.automatic_failure_candidates,
                review_id=latest.review_id,
                review_revision_count=len(revisions),
                review_verdict=latest.verdict.value,
                confirmed_failure_types=latest.confirmed_failure_types,
                review_notes=latest.notes,
            )
        )

    summary = HumanReviewSummary(
        reviewed_prediction_count=len(reviewed_cases),
        review_revision_count=sum(
            len(reviews_by_prediction[prediction.prediction_id]) for prediction in predictions
        ),
        confirmed_count=sum(review.verdict is ReviewVerdict.CONFIRMED for review in latest_reviews),
        rejected_count=sum(review.verdict is ReviewVerdict.REJECTED for review in latest_reviews),
        uncertain_count=sum(review.verdict is ReviewVerdict.UNCERTAIN for review in latest_reviews),
        confirmed_failure_type_counts=tuple(
            FailureTypeCount(failure_type=name, count=count)
            for name, count in sorted(failure_counts.items())
        ),
        confirmed_language_model_bias_count=failure_counts["LANGUAGE_MODEL_BIAS"],
    )
    limitations = _limitations(source, summary)
    report = HumanReviewedBaselineReport(
        report_id=f"{source.report_id}--human-reviewed",
        source_report_id=source.report_id,
        experiment_id=experiment_id,
        benchmark_id=source.benchmark_id,
        benchmark_version=source.benchmark_version,
        created_at=max(source.created_at, *(review.created_at for review in latest_reviews)),
        model=source.model,
        automatic_metrics=source.metrics,
        human_review=summary,
        cases=tuple(reviewed_cases),
        limitations=limitations,
        gate_decision="baseline_established_with_limitations",
    )

    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "-", experiment_id).strip(".-")
    stem = f"{safe_id or 'experiment'}--reviewed"
    json_path = storage.reports_dir / f"{stem}.json"
    markdown_path = storage.reports_dir / f"{stem}.md"
    _write(json_path, report.model_dump_json(indent=2) + "\n")
    _write(markdown_path, _render_markdown(report))
    return report, json_path, markdown_path


def _load_source_report(storage: LabStorage, experiment_id: str) -> BaselineReport:
    reports = tuple(
        BaselineReport.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(storage.reports_dir.glob("report-*.json"))
    )
    matches = tuple(report for report in reports if report.experiment_id == experiment_id)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one automatic baseline report for {experiment_id}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _validate_review(review: HumanReview, prediction: StoredPrediction) -> None:
    if review.utterance_id != prediction.utterance_id:
        raise ValueError("review utterance does not match its prediction")
    unknown = set(review.confirmed_failure_types) - VALID_FAILURE_TYPES
    if unknown:
        raise ValueError(f"unknown confirmed failure type: {', '.join(sorted(unknown))}")
    if review.verdict is not ReviewVerdict.CONFIRMED and review.confirmed_failure_types:
        raise ValueError("only confirmed reviews may contain confirmed failure types")
    if review.verdict is ReviewVerdict.UNCERTAIN and not review.notes:
        raise ValueError("uncertain reviews require an explanatory note")


def _limitations(
    source: BaselineReport,
    summary: HumanReviewSummary,
) -> tuple[str, ...]:
    items = [
        "방언 자동 판정은 정확한 문자열 포함 여부를 사용해 자연스러운 변이형을 "
        "완전히 구분하지 못한다.",
    ]
    if all(case.confidence is None for case in source.cases):
        items.append("이 런타임은 신뢰할 수 있는 confidence를 제공하지 않았다.")
    if summary.uncertain_count:
        items.append(f"사람 검수 {summary.uncertain_count}건은 불확실로 남아 있다.")
    if (
        source.metrics.context_overcorrection_rate == 0
        and summary.confirmed_language_model_bias_count
    ):
        items.append(
            "자동 과보정률은 0이지만 사람이 LANGUAGE_MODEL_BIAS를 "
            f"{summary.confirmed_language_model_bias_count}건 확인했다."
        )
    if source.model.checkpoint_identifier is None:
        items.append("정확한 checkpoint_identifier와 호환 모델 런타임의 독립 검증이 남아 있다.")
    return tuple(items)


def _render_markdown(report: HumanReviewedBaselineReport) -> str:
    metrics = report.automatic_metrics
    review = report.human_review
    checkpoint = report.model.checkpoint_identifier or report.model.checkpoint or "미기록"
    failure_counts = (
        ", ".join(
            f"{item.failure_type} {item.count}건" for item in review.confirmed_failure_type_counts
        )
        or "없음"
    )
    lines = [
        _report_title(report.experiment_id),
        "",
        "## 실험 조건",
        "",
        f"- Experiment: `{report.experiment_id}`",
        f"- Benchmark: `{report.benchmark_id}@{report.benchmark_version}`",
        f"- Model: `{report.model.name}` / `{report.model.version}`",
        f"- Checkpoint: `{checkpoint}`",
        f"- Source report: `{report.source_report_id}`",
        "",
        "## 자동 평가",
        "",
        f"- Utterances: {metrics.utterance_count}",
        f"- CER: {metrics.cer:.4f}",
        f"- Dialect preservation: {metrics.dialect_preservation_rate:.4f}",
        f"- Context overcorrection: {metrics.context_overcorrection_rate:.4f}",
        f"- High-confidence wrong: {metrics.high_confidence_wrong_count}",
        f"- Latency p50/p95: {metrics.p50_latency_ms:.2f} / {metrics.p95_latency_ms:.2f} ms",
        "",
        "## 사람 검수",
        "",
        f"- Latest reviews: {review.reviewed_prediction_count}",
        f"- All revisions: {review.review_revision_count}",
        f"- Confirmed / Rejected / Uncertain: "
        f"{review.confirmed_count} / {review.rejected_count} / {review.uncertain_count}",
        f"- Confirmed failure types: {failure_counts}",
        "",
        "## 발화별 결과",
        "",
        "| Surface 정답 | 모델 예측 | CER | 자동 방언 판정 | 사람 판정 | 오류 유형 | 검수 메모 |",
        "|---|---|---:|---|---|---|---|",
    ]
    for case in report.cases:
        dialect = (
            ", ".join(
                f"{result.surface_form}:{result.match_status.value}"
                for result in case.dialect.results
            )
            or "표현 없음"
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    _cell(case.reference_surface_text),
                    _cell(case.hypothesis_surface_text),
                    f"{case.cer:.3f}",
                    _cell(dialect),
                    case.review_verdict,
                    _cell(", ".join(case.confirmed_failure_types) or "없음"),
                    _cell(case.review_notes or ""),
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
            "## Gate 1 판정",
            "",
            f"`{report.gate_decision}`",
            "",
            "Baseline과 평가 계약은 확립됐으며, 확인된 실패를 기준으로 "
            "Surface ASR 개선이 필요하다.",
            "",
        ]
    )
    return "\n".join(lines)


def _report_title(experiment_id: str) -> str:
    match = re.match(r"^(task-\d+[a-z]?)", experiment_id, flags=re.IGNORECASE)
    if match is None:
        return "# Human-reviewed Surface ASR Baseline Report"
    return f"# {match.group(1).upper()} Human-reviewed Baseline Report"


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def _write(path: Path, content: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
