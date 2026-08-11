"""Transparent metrics for surface transcription and dialect preservation."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from statistics import median
from uuid import UUID

import numpy as np

from busan_lab.schemas.asr import ModelDescriptor
from busan_lab.schemas.evaluation import (
    AggregateMetrics,
    DialectExpressionResult,
    DialectMatchStatus,
    DialectPreservationMetric,
    EditCounts,
    EvaluationCaseResult,
)
from busan_lab.schemas.utterance import DialectExpressionLabel

_IGNORED_CATEGORIES = {"Z", "P"}


def normalize_for_cer(text: str) -> str:
    """NFC-normalize and remove whitespace/punctuation without changing words."""

    normalized = unicodedata.normalize("NFC", text)
    return "".join(
        character
        for character in normalized
        if unicodedata.category(character)[0] not in _IGNORED_CATEGORIES
    )


def character_error_rate(reference: str, hypothesis: str) -> tuple[float, EditCounts]:
    normalized_reference = normalize_for_cer(reference)
    normalized_hypothesis = normalize_for_cer(hypothesis)
    edits = _levenshtein_counts(normalized_reference, normalized_hypothesis)
    if edits.reference_characters == 0:
        cer = 0.0 if not normalized_hypothesis else float(edits.insertions)
    else:
        cer = (
            edits.substitutions + edits.deletions + edits.insertions
        ) / edits.reference_characters
    return cer, edits


def evaluate_dialect_preservation(
    hypothesis: str,
    labels: Sequence[DialectExpressionLabel],
) -> DialectPreservationMetric:
    normalized_hypothesis = unicodedata.normalize("NFC", hypothesis)
    results: list[DialectExpressionResult] = []
    preserved = 0
    overcorrected = 0
    approved = 0

    for label in labels:
        surface = unicodedata.normalize("NFC", label.surface_form)
        normalized_forms = tuple(
            unicodedata.normalize("NFC", value) for value in label.normalized_forms
        )
        if label.status.value == "approved":
            approved += 1
        if surface in normalized_hypothesis:
            match_status = DialectMatchStatus.PRESERVED
            matched_form: str | None = label.surface_form
            preserved += 1
        else:
            matched_form = next(
                (value for value in normalized_forms if value in normalized_hypothesis),
                None,
            )
            if matched_form is not None:
                match_status = DialectMatchStatus.OVERCORRECTED
                overcorrected += 1
            else:
                match_status = DialectMatchStatus.MISSING
        results.append(
            DialectExpressionResult(
                surface_form=label.surface_form,
                normalized_forms=label.normalized_forms,
                label_status=label.status,
                match_status=match_status,
                matched_form=matched_form,
            )
        )

    total = len(labels)
    return DialectPreservationMetric(
        preservation_rate=preserved / total if total else 1.0,
        overcorrection_rate=overcorrected / total if total else 0.0,
        reference_expression_count=total,
        approved_expression_count=approved,
        results=tuple(results),
    )


def evaluate_case(
    *,
    utterance_id: UUID,
    reference_surface_text: str,
    normalized_meaning: str | None,
    hypothesis_surface_text: str,
    confidence: float | None,
    dialect_labels: Sequence[DialectExpressionLabel],
    model: ModelDescriptor,
    high_confidence_threshold: float = 0.85,
) -> EvaluationCaseResult:
    cer, edits = character_error_rate(reference_surface_text, hypothesis_surface_text)
    dialect = evaluate_dialect_preservation(hypothesis_surface_text, dialect_labels)
    return EvaluationCaseResult(
        utterance_id=utterance_id,
        reference_surface_text=reference_surface_text,
        normalized_meaning=normalized_meaning,
        hypothesis_surface_text=hypothesis_surface_text,
        confidence=confidence,
        cer=cer,
        edits=edits,
        dialect=dialect,
        high_confidence_wrong=(
            cer > 0
            and confidence is not None
            and confidence >= high_confidence_threshold
        ),
        model=model,
    )


def aggregate_cases(
    cases: Sequence[EvaluationCaseResult],
    latencies_ms: Sequence[float],
) -> AggregateMetrics:
    if len(cases) != len(latencies_ms):
        raise ValueError("every evaluation case requires one latency")
    if not cases:
        return AggregateMetrics(
            utterance_count=0,
            cer=0,
            dialect_preservation_rate=1,
            context_overcorrection_rate=0,
            high_confidence_wrong_count=0,
            p50_latency_ms=0,
            p95_latency_ms=0,
        )

    total_edits = sum(
        case.edits.substitutions + case.edits.deletions + case.edits.insertions for case in cases
    )
    total_reference = sum(case.edits.reference_characters for case in cases)
    expression_count = sum(case.dialect.reference_expression_count for case in cases)
    preserved_count = sum(
        result.match_status is DialectMatchStatus.PRESERVED
        for case in cases
        for result in case.dialect.results
    )
    overcorrected_count = sum(
        result.match_status is DialectMatchStatus.OVERCORRECTED
        for case in cases
        for result in case.dialect.results
    )
    latency_array = np.asarray(latencies_ms, dtype=np.float64)
    return AggregateMetrics(
        utterance_count=len(cases),
        cer=total_edits / total_reference if total_reference else float(total_edits),
        dialect_preservation_rate=(preserved_count / expression_count if expression_count else 1),
        context_overcorrection_rate=(
            overcorrected_count / expression_count if expression_count else 0
        ),
        high_confidence_wrong_count=sum(case.high_confidence_wrong for case in cases),
        p50_latency_ms=float(median(latencies_ms)),
        p95_latency_ms=float(np.percentile(latency_array, 95)),
    )


def _levenshtein_counts(reference: str, hypothesis: str) -> EditCounts:
    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    costs = np.zeros((rows, columns), dtype=np.int32)
    operations = np.zeros((rows, columns), dtype=np.int8)
    costs[:, 0] = np.arange(rows)
    costs[0, :] = np.arange(columns)
    operations[1:, 0] = 2  # deletion
    operations[0, 1:] = 3  # insertion

    for row in range(1, rows):
        for column in range(1, columns):
            if reference[row - 1] == hypothesis[column - 1]:
                diagonal_cost = int(costs[row - 1, column - 1])
                diagonal_operation = 0
            else:
                diagonal_cost = int(costs[row - 1, column - 1]) + 1
                diagonal_operation = 1  # substitution
            candidates = (
                (diagonal_cost, diagonal_operation),
                (int(costs[row - 1, column]) + 1, 2),
                (int(costs[row, column - 1]) + 1, 3),
            )
            best_cost, best_operation = min(candidates, key=lambda candidate: candidate[0])
            costs[row, column] = best_cost
            operations[row, column] = best_operation

    substitutions = deletions = insertions = 0
    row = len(reference)
    column = len(hypothesis)
    while row or column:
        operation = int(operations[row, column])
        if operation == 0:
            row -= 1
            column -= 1
        elif operation == 1:
            substitutions += 1
            row -= 1
            column -= 1
        elif operation == 2:
            deletions += 1
            row -= 1
        else:
            insertions += 1
            column -= 1

    return EditCounts(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_characters=len(reference),
    )
