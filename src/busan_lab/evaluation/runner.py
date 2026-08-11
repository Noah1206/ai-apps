"""Run a fixed benchmark through a replaceable Surface ASR adapter."""

from __future__ import annotations

from uuid import uuid4

from busan_lab.adapters.base import SurfaceASRAdapter
from busan_lab.evaluation.metrics import aggregate_cases, evaluate_case
from busan_lab.schemas.benchmark import BenchmarkManifest
from busan_lab.schemas.evaluation import BaselineReport, ErrorExportRecord
from busan_lab.schemas.experiment import (
    ExperimentRun,
    PredictionSource,
    StoredPrediction,
)
from busan_lab.storage import LabStorage


class BenchmarkRunner:
    def __init__(self, storage: LabStorage) -> None:
        self.storage = storage

    def run(
        self,
        manifest: BenchmarkManifest,
        adapter: SurfaceASRAdapter,
        *,
        export_failures: bool = True,
    ) -> BaselineReport:
        experiment_id = adapter.experiment_id
        self.storage.save_experiment(
            ExperimentRun(
                experiment_id=experiment_id,
                model=adapter.model,
                benchmark_id=manifest.benchmark_id,
                benchmark_version=manifest.benchmark_version,
            )
        )
        cases = []
        latencies = []
        for entry in manifest.entries:
            audio_path = self.storage.resolve(entry.derived_audio_path)
            result = adapter.transcribe(
                audio_path,
                utterance_id=str(entry.utterance_id),
                audio_sha256=entry.derived_audio_sha256,
            )
            case = evaluate_case(
                utterance_id=entry.utterance_id,
                reference_surface_text=entry.surface_text,
                normalized_meaning=entry.normalized_meaning,
                hypothesis_surface_text=result.surface_text,
                confidence=result.confidence,
                dialect_labels=entry.dialect_expressions,
                model=result.model,
            )
            if export_failures:
                taxonomies: list[str] = []
                if case.dialect.overcorrection_rate > 0:
                    taxonomies.extend(("LANGUAGE_MODEL_BIAS", "DECODING"))
                if case.high_confidence_wrong:
                    taxonomies.append("CALIBRATION")
                if taxonomies:
                    self.storage.append_error(
                        ErrorExportRecord(
                            failure_taxonomy=tuple(dict.fromkeys(taxonomies)),
                            case=case,
                        )
                    )
            else:
                taxonomies = []
                if case.dialect.overcorrection_rate > 0:
                    taxonomies.extend(("LANGUAGE_MODEL_BIAS", "DECODING"))
                if case.high_confidence_wrong:
                    taxonomies.append("CALIBRATION")
            self.storage.save_prediction(
                StoredPrediction(
                    experiment_id=experiment_id,
                    utterance_id=entry.utterance_id,
                    audio_sha256=entry.derived_audio_sha256,
                    source=PredictionSource.PRECOMPUTED,
                    latency_ms=result.latency_ms,
                    automatic_failure_candidates=tuple(dict.fromkeys(taxonomies)),
                    evaluation=case,
                )
            )
            cases.append(case)
            latencies.append(result.latency_ms)

        report = BaselineReport(
            report_id=f"report-{uuid4()}",
            experiment_id=experiment_id,
            benchmark_id=manifest.benchmark_id,
            benchmark_version=manifest.benchmark_version,
            model=adapter.model,
            metrics=aggregate_cases(cases, latencies),
            cases=tuple(cases),
        )
        report_path = self.storage.reports_dir / f"{report.report_id}.json"
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(report_path)
        return report
