"""Command-line entry points for serving and fixed benchmark evaluation."""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

import uvicorn

from busan_lab.adapters.precomputed import PrecomputedSurfaceASRAdapter
from busan_lab.audio import AudioProcessor, hash_file
from busan_lab.config import LabSettings
from busan_lab.environment import find_project_root, inspect_project_environment
from busan_lab.evaluation.calibration import (
    load_calibration_profile,
    recalibrate_experiment,
)
from busan_lab.evaluation.reporting import finalize_human_reviewed_report
from busan_lab.evaluation.runner import BenchmarkRunner
from busan_lab.schemas import (
    BenchmarkIntegrityAudit,
    BenchmarkManifest,
    BlindABReviewResult,
    EvaluationCalibrationProfile,
    EvaluationCalibrationReport,
    EvaluationCaseResult,
    EvaluationExclusionRegistry,
    ExperimentRun,
    Gate2Assessment,
    Gate2Criteria,
    Gate2EvaluationManifest,
    Gate2Evidence,
    HumanReview,
    HumanReviewedBaselineReport,
    LabelRevision,
    PrecomputedPrediction,
    PredictionComparison,
    ReproducibilitySpec,
    StoredPrediction,
    TrainingDatasetManifest,
    TrainingDatasetValidationReport,
    TrainingExportRecord,
    TrainingRecordingImportManifest,
    TrainingRecordingImportPlan,
    TrainingSplitAssignments,
    UtteranceRecord,
)
from busan_lab.schemas.common import ConsentRecord, ReviewStatus
from busan_lab.storage import LabStorage, file_is_dataless
from busan_lab.training import (
    build_training_dataset,
    export_training_dataset_bundle,
    review_training_label,
    validate_training_dataset,
)
from busan_lab.training_import import (
    build_training_recording_import_plan,
    execute_training_recording_import,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="busan-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Run the local research UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    schemas = subparsers.add_parser("export-schemas", help="Export versioned JSON schemas")
    schemas.add_argument("--output", type=Path, default=Path("reports/schemas"))

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate precomputed Surface ASR predictions",
    )
    evaluate.add_argument("--benchmark-id", required=True)
    evaluate.add_argument("--benchmark-version", required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--data-root", type=Path, default=None)

    export_benchmark = subparsers.add_parser(
        "export-benchmark",
        help="Bundle one frozen benchmark and its derived WAV files for GPU inference",
    )
    export_benchmark.add_argument("--benchmark-id", required=True)
    export_benchmark.add_argument("--benchmark-version", required=True)
    export_benchmark.add_argument("--output", type=Path, required=True)
    export_benchmark.add_argument("--data-root", type=Path, default=None)

    review_training = subparsers.add_parser(
        "review-training-label",
        help="Append a human-reviewed or approved label revision for training",
    )
    review_training.add_argument("--utterance-id", required=True, type=UUID)
    review_training.add_argument("--reviewer-id", required=True)
    review_training.add_argument(
        "--status",
        required=True,
        choices=(
            ReviewStatus.HUMAN_REVIEWED.value,
            ReviewStatus.APPROVED.value,
        ),
    )
    review_training.add_argument("--reason", default=None)
    review_training.add_argument("--data-root", type=Path, default=None)

    create_training = subparsers.add_parser(
        "create-training-dataset",
        help="Create a frozen, leakage-checked train/validation manifest",
    )
    create_training.add_argument("--dataset-id", required=True)
    create_training.add_argument("--dataset-version", required=True)
    create_training.add_argument("--assignments", required=True, type=Path)
    create_training.add_argument("--data-root", type=Path, default=None)

    validate_training = subparsers.add_parser(
        "validate-training-dataset",
        help="Revalidate a training manifest against all frozen benchmarks",
    )
    validate_training.add_argument("--dataset-id", required=True)
    validate_training.add_argument("--dataset-version", required=True)
    validate_training.add_argument("--data-root", type=Path, default=None)

    export_training = subparsers.add_parser(
        "export-training-dataset",
        help="Export a validated model-neutral training ZIP",
    )
    export_training.add_argument("--dataset-id", required=True)
    export_training.add_argument("--dataset-version", required=True)
    export_training.add_argument("--output", required=True, type=Path)
    export_training.add_argument("--data-root", type=Path, default=None)

    import_recordings = subparsers.add_parser(
        "import-training-recordings",
        help="Validate or import a numbered TASK-004 M4A recording batch",
    )
    import_recordings.add_argument("--import-id", required=True)
    import_recordings.add_argument("--input-dir", required=True, type=Path)
    import_recordings.add_argument("--prompt-sheet", required=True, type=Path)
    import_recordings.add_argument("--prompt-start", type=int, default=1)
    import_recordings.add_argument("--prompt-end", type=int, required=True)
    import_recordings.add_argument("--speaker-id", required=True)
    import_recordings.add_argument("--region", default="Busan")
    import_recordings.add_argument("--device", required=True)
    import_recordings.add_argument("--recording-environment", required=True)
    import_recordings.add_argument("--confirm-storage-consent", action="store_true")
    import_recordings.add_argument("--confirm-research-use", action="store_true")
    import_recordings.add_argument(
        "--confirm-model-training-consent",
        action="store_true",
    )
    import_recordings.add_argument(
        "--commit",
        action="store_true",
        help="Persist converted audio, candidate records, and the import ledger",
    )
    import_recordings.add_argument("--data-root", type=Path, default=None)

    finalize_report = subparsers.add_parser(
        "finalize-report",
        help="Merge one automatic baseline with the latest human reviews",
    )
    finalize_report.add_argument("--experiment-id", required=True)
    finalize_report.add_argument("--data-root", type=Path, default=None)

    calibrate_evaluation = subparsers.add_parser(
        "calibrate-evaluation",
        help="Re-evaluate stored predictions and reviews with a versioned calibration profile",
    )
    calibrate_evaluation.add_argument("--experiment-id", required=True)
    calibrate_evaluation.add_argument("--profile", type=Path, required=True)
    calibrate_evaluation.add_argument(
        "--next-experiment",
        default=None,
        help="Optional planning note for the generated Markdown report",
    )
    calibrate_evaluation.add_argument("--data-root", type=Path, default=None)

    subparsers.add_parser(
        "doctor",
        help="Verify the pinned project-local Python and audio tool environment",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(arguments)
    if parsed.command == "serve":
        uvicorn.run(
            "busan_lab.api:app",
            host=parsed.host,
            port=parsed.port,
            reload=parsed.reload,
        )
        return 0
    if parsed.command == "export-schemas":
        export_schemas(parsed.output)
        return 0
    if parsed.command == "evaluate":
        settings = LabSettings.from_environment(parsed.data_root)
        storage = LabStorage(settings.data_root)
        evaluation_manifest = storage.load_manifest(
            parsed.benchmark_id,
            parsed.benchmark_version,
        )
        adapter = PrecomputedSurfaceASRAdapter(
            parsed.predictions,
            evaluation_manifest,
        )
        evaluation_report = BenchmarkRunner(storage).run(
            evaluation_manifest,
            adapter,
        )
        print(evaluation_report.model_dump_json(indent=2))
        return 0
    if parsed.command == "export-benchmark":
        settings = LabSettings.from_environment(parsed.data_root)
        storage = LabStorage(settings.data_root)
        export_manifest = storage.load_manifest(
            parsed.benchmark_id,
            parsed.benchmark_version,
        )
        export_benchmark_bundle(storage, export_manifest, parsed.output)
        print(parsed.output.expanduser().resolve())
        return 0
    if parsed.command == "review-training-label":
        settings = LabSettings.from_environment(parsed.data_root)
        record = review_training_label(
            LabStorage(settings.data_root),
            utterance_id=parsed.utterance_id,
            reviewer_id=parsed.reviewer_id,
            status=ReviewStatus(parsed.status),
            reason=parsed.reason,
        )
        print(record.model_dump_json(indent=2))
        return 0
    if parsed.command == "create-training-dataset":
        settings = LabSettings.from_environment(parsed.data_root)
        storage = LabStorage(settings.data_root)
        assignments = TrainingSplitAssignments.model_validate(
            json.loads(parsed.assignments.read_text(encoding="utf-8"))
        )
        training_manifest, training_report = build_training_dataset(
            dataset_id=parsed.dataset_id,
            dataset_version=parsed.dataset_version,
            records=storage.list_utterances(),
            assignments=assignments,
            benchmark_manifests=storage.list_manifests(),
        )
        training_path = storage.save_training_dataset(training_manifest)
        print(training_manifest.model_dump_json(indent=2))
        print(training_report.model_dump_json(indent=2))
        print(training_path)
        return 0
    if parsed.command == "validate-training-dataset":
        settings = LabSettings.from_environment(parsed.data_root)
        storage = LabStorage(settings.data_root)
        training_manifest_to_validate = storage.load_training_dataset(
            parsed.dataset_id,
            parsed.dataset_version,
        )
        validation_report = validate_training_dataset(
            training_manifest_to_validate,
            storage.list_manifests(),
        )
        print(validation_report.model_dump_json(indent=2))
        return 0 if validation_report.passed else 1
    if parsed.command == "export-training-dataset":
        settings = LabSettings.from_environment(parsed.data_root)
        storage = LabStorage(settings.data_root)
        training_manifest_to_export = storage.load_training_dataset(
            parsed.dataset_id,
            parsed.dataset_version,
        )
        export_validation_report = validate_training_dataset(
            training_manifest_to_export,
            storage.list_manifests(),
        )
        training_export_path = export_training_dataset_bundle(
            storage,
            training_manifest_to_export,
            export_validation_report,
            parsed.output,
        )
        print(training_export_path)
        return 0
    if parsed.command == "import-training-recordings":
        settings = LabSettings.from_environment(parsed.data_root)
        storage = LabStorage(settings.data_root)
        import_plan = build_training_recording_import_plan(
            import_id=parsed.import_id,
            input_directory=parsed.input_dir,
            prompt_sheet=parsed.prompt_sheet,
            prompt_start=parsed.prompt_start,
            prompt_end=parsed.prompt_end,
            speaker_id=parsed.speaker_id,
            region=parsed.region,
            device=parsed.device,
            recording_environment=parsed.recording_environment,
            consent=ConsentRecord(
                storage_allowed=parsed.confirm_storage_consent,
                research_use_allowed=parsed.confirm_research_use,
                model_training_allowed=parsed.confirm_model_training_consent,
            ),
            benchmark_manifests=storage.list_manifests(),
        )
        print(import_plan.model_dump_json(indent=2))
        if not import_plan.passed:
            return 1
        if not parsed.commit:
            return 0
        import_manifest = execute_training_recording_import(
            plan=import_plan,
            input_directory=parsed.input_dir,
            storage=storage,
            processor=AudioProcessor(settings, storage),
        )
        print(import_manifest.model_dump_json(indent=2))
        return 0
    if parsed.command == "finalize-report":
        settings = LabSettings.from_environment(parsed.data_root)
        reviewed_report, json_path, markdown_path = finalize_human_reviewed_report(
            LabStorage(settings.data_root),
            parsed.experiment_id,
        )
        print(reviewed_report.model_dump_json(indent=2))
        print(json_path)
        print(markdown_path)
        return 0
    if parsed.command == "calibrate-evaluation":
        settings = LabSettings.from_environment(parsed.data_root)
        calibration_report, json_path, markdown_path = recalibrate_experiment(
            LabStorage(settings.data_root),
            parsed.experiment_id,
            load_calibration_profile(parsed.profile),
            next_experiment=parsed.next_experiment,
        )
        print(calibration_report.model_dump_json(indent=2))
        print(json_path)
        print(markdown_path)
        return 0
    if parsed.command == "doctor":
        environment_report = inspect_project_environment(find_project_root())
        print(environment_report.model_dump_json(indent=2))
        return 0 if environment_report.passed else 1
    raise AssertionError(f"unknown command: {parsed.command}")


def export_benchmark_bundle(
    storage: LabStorage,
    manifest: BenchmarkManifest,
    output_path: Path,
) -> Path:
    """Write the exact frozen manifest and hash-verified derived WAV files."""

    if not manifest.frozen:
        raise ValueError("only frozen benchmarks can be exported")
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("benchmark.json", manifest.model_dump_json(indent=2))
            for entry in manifest.entries:
                audio_path = storage.resolve(entry.derived_audio_path)
                if not audio_path.is_file():
                    raise FileNotFoundError(f"benchmark audio is missing: {audio_path}")
                if file_is_dataless(audio_path):
                    raise OSError(
                        f"benchmark audio is cloud-only: {audio_path}. "
                        "Download data/lab to this Mac and retry."
                    )
                if hash_file(audio_path) != entry.derived_audio_sha256:
                    raise ValueError(f"benchmark audio hash mismatch: {audio_path}")
                archive.write(audio_path, entry.derived_audio_path)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def export_schemas(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    schemas = {
        "utterance.schema.json": UtteranceRecord.model_json_schema(),
        "benchmark-manifest.schema.json": BenchmarkManifest.model_json_schema(),
        "benchmark-integrity-audit.schema.json": BenchmarkIntegrityAudit.model_json_schema(),
        "gate2-blind-ab-review.schema.json": BlindABReviewResult.model_json_schema(),
        "gate2-evaluation-dataset.schema.json": Gate2EvaluationManifest.model_json_schema(),
        "gate2-evaluation-exclusions.schema.json": (
            EvaluationExclusionRegistry.model_json_schema()
        ),
        "gate2-reproducibility-spec.schema.json": ReproducibilitySpec.model_json_schema(),
        "gate2-criteria.schema.json": Gate2Criteria.model_json_schema(),
        "gate2-evidence.schema.json": Gate2Evidence.model_json_schema(),
        "gate2-assessment.schema.json": Gate2Assessment.model_json_schema(),
        "evaluation-case.schema.json": EvaluationCaseResult.model_json_schema(),
        "experiment-run.schema.json": ExperimentRun.model_json_schema(),
        "stored-prediction.schema.json": StoredPrediction.model_json_schema(),
        "human-review.schema.json": HumanReview.model_json_schema(),
        "label-revision.schema.json": LabelRevision.model_json_schema(),
        "prediction-comparison.schema.json": PredictionComparison.model_json_schema(),
        "precomputed-prediction.schema.json": PrecomputedPrediction.model_json_schema(),
        "human-reviewed-baseline.schema.json": (HumanReviewedBaselineReport.model_json_schema()),
        "evaluation-calibration-profile.schema.json": (
            EvaluationCalibrationProfile.model_json_schema()
        ),
        "evaluation-calibration-report.schema.json": (
            EvaluationCalibrationReport.model_json_schema()
        ),
        "training-dataset.schema.json": TrainingDatasetManifest.model_json_schema(),
        "training-dataset-validation-report.schema.json": (
            TrainingDatasetValidationReport.model_json_schema()
        ),
        "training-export-record.schema.json": TrainingExportRecord.model_json_schema(),
        "training-recording-import-manifest.schema.json": (
            TrainingRecordingImportManifest.model_json_schema()
        ),
        "training-recording-import-plan.schema.json": (
            TrainingRecordingImportPlan.model_json_schema()
        ),
        "training-split-assignments.schema.json": (
            TrainingSplitAssignments.model_json_schema()
        ),
    }
    for filename, schema in schemas.items():
        (output_dir / filename).write_text(
            json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    raise SystemExit(main())
