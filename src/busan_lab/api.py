"""FastAPI surface for the offline research lab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, cast
from uuid import UUID, uuid4

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field, ValidationError

from busan_lab.analysis import analyze_audio
from busan_lab.audio import AudioProcessor, AudioValidationError
from busan_lab.config import LabSettings
from busan_lab.evaluation.calibration import diagnose_prediction
from busan_lab.evaluation.metrics import evaluate_case
from busan_lab.manifest import build_manifest
from busan_lab.schemas.asr import ModelDescriptor
from busan_lab.schemas.audio import AudioRole
from busan_lab.schemas.benchmark import BenchmarkManifest
from busan_lab.schemas.calibration import PredictionDiagnostics
from busan_lab.schemas.common import ConsentRecord, DatasetSplit, StrictSchema
from busan_lab.schemas.evaluation import ErrorExportRecord, EvaluationCaseResult
from busan_lab.schemas.experiment import (
    ExperimentRun,
    HumanReview,
    PredictionComparison,
    PredictionSource,
    ReviewVerdict,
    StoredPrediction,
)
from busan_lab.schemas.training import TrainingDatasetManifest
from busan_lab.schemas.training_import import (
    TrainingRecordingImportManifest,
    TrainingRecordingImportPlan,
    TrainingRecordingImportSummary,
    TrainingRecordingReviewQueue,
    TrainingRecordingReviewRequest,
)
from busan_lab.schemas.utterance import (
    DialectExpressionLabel,
    LabelRevision,
    LinguisticGroundTruth,
    SpeakerContext,
    UtteranceRecord,
    expand_dialect_expression_labels,
)
from busan_lab.storage import (
    DatalessFileError,
    LabStorage,
    RecordNotFoundError,
    file_is_dataless,
)
from busan_lab.training_import import (
    build_training_recording_review_queue,
    list_training_recording_import_summaries,
    review_training_recording,
)

STATIC_DIR = Path(__file__).parent / "static"


class ManualEvaluationRequest(StrictSchema):
    utterance_id: UUID
    hypothesis_surface_text: str
    confidence: float = Field(ge=0, le=1)
    model: ModelDescriptor


class CreatePredictionRequest(ManualEvaluationRequest):
    experiment_id: str = Field(min_length=1, max_length=128)
    latency_ms: float | None = Field(default=None, ge=0)


class CreateHumanReviewRequest(StrictSchema):
    prediction_id: UUID
    reviewer_id: str = Field(min_length=1, max_length=128)
    verdict: ReviewVerdict
    confirmed_failure_types: tuple[str, ...] = ()
    notes: str | None = Field(default=None, max_length=4000)


class ComparePredictionsRequest(StrictSchema):
    prediction_a_id: UUID
    prediction_b_id: UUID


class CreateBenchmarkRequest(StrictSchema):
    benchmark_id: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    utterance_ids: tuple[UUID, ...]
    split: DatasetSplit = DatasetSplit.TEST


class UpdateLabelRequest(StrictSchema):
    surface_text: str = Field(min_length=1, max_length=10_000)
    normalized_meaning: str | None = Field(default=None, max_length=10_000)
    dialect_expressions: tuple[DialectExpressionLabel, ...] = ()
    changed_by: str = Field(min_length=1, max_length=128)
    reason: str | None = Field(default=None, max_length=4000)


class DeleteUtteranceResponse(StrictSchema):
    utterance_id: UUID
    archive_id: str
    archived_paths: tuple[str, ...]
    preserved_shared_paths: tuple[str, ...]
    removed_export_rows: int = Field(ge=0)


class DeleteBenchmarkResponse(StrictSchema):
    benchmark_id: str
    benchmark_version: str
    archive_id: str
    archived_path: str
    entry_count: int = Field(ge=0)


def create_app(data_root: Path | None = None) -> FastAPI:
    settings = LabSettings.from_environment(data_root)
    storage = LabStorage(settings.data_root)
    processor = AudioProcessor(settings, storage)
    application = FastAPI(
        title="Busan Speech Research Lab v0",
        version="0.1.0",
        description="Measurement foundation for Surface ASR research.",
    )
    application.state.settings = settings
    application.state.storage = storage
    application.state.processor = processor
    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @application.exception_handler(DatalessFileError)
    async def dataless_file_handler(_request, error: DatalessFileError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "detail": (
                    "Record data is stored in cloud-only mode. "
                    "Download data/lab to this Mac and retry. "
                    f"({error.path.name})"
                ),
            },
        )

    def evaluate_and_store(
        *,
        request: ManualEvaluationRequest,
        experiment_id: str,
        source: PredictionSource,
        latency_ms: float | None = None,
    ) -> StoredPrediction:
        record = storage.load_utterance(request.utterance_id)
        experiment = ExperimentRun(
            experiment_id=experiment_id,
            model=request.model,
        )
        storage.save_experiment(experiment)
        case = evaluate_case(
            utterance_id=record.utterance_id,
            reference_surface_text=record.ground_truth.surface_text,
            normalized_meaning=record.ground_truth.normalized_meaning,
            hypothesis_surface_text=request.hypothesis_surface_text,
            confidence=request.confidence,
            dialect_labels=record.ground_truth.dialect_expressions,
            model=request.model,
        )
        failure_candidates: list[str] = []
        if case.dialect.overcorrection_rate > 0:
            failure_candidates.extend(("LANGUAGE_MODEL_BIAS", "DECODING"))
        if case.high_confidence_wrong:
            failure_candidates.append("CALIBRATION")
        prediction = StoredPrediction(
            experiment_id=experiment_id,
            utterance_id=record.utterance_id,
            audio_sha256=record.audio.derived.sha256,
            source=source,
            latency_ms=latency_ms,
            automatic_failure_candidates=tuple(dict.fromkeys(failure_candidates)),
            evaluation=case,
        )
        storage.save_prediction(prediction)
        if failure_candidates:
            storage.append_error(
                ErrorExportRecord(
                    failure_taxonomy=prediction.automatic_failure_candidates,
                    case=case,
                )
            )
        return prediction

    @application.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @application.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "phase": "measurement_foundation", "task": "TASK-001"}

    @application.get("/api/utterances", response_model=list[UtteranceRecord])
    async def list_utterances() -> list[UtteranceRecord]:
        return list(storage.list_utterances())

    @application.get(
        "/api/training-imports",
        response_model=list[TrainingRecordingImportSummary],
    )
    async def list_training_imports() -> list[TrainingRecordingImportSummary]:
        return list(list_training_recording_import_summaries(storage))

    @application.get(
        "/api/training-imports/{import_id}/review-queue",
        response_model=TrainingRecordingReviewQueue,
    )
    async def get_training_recording_review_queue(
        import_id: str,
    ) -> TrainingRecordingReviewQueue:
        try:
            return build_training_recording_review_queue(storage, import_id=import_id)
        except RecordNotFoundError as error:
            raise HTTPException(
                status_code=404,
                detail=f"Training recording import or utterance not found: {error.args[0]}",
            ) from error

    @application.post(
        "/api/training-imports/{import_id}/review-queue/{prompt_id}/decision",
        response_model=UtteranceRecord,
    )
    async def decide_training_recording_review(
        import_id: str,
        prompt_id: str,
        request: TrainingRecordingReviewRequest,
    ) -> UtteranceRecord:
        try:
            return review_training_recording(
                storage,
                import_id=import_id,
                prompt_id=prompt_id,
                reviewer_id=request.reviewer_id,
                decision=request.decision,
                notes=request.notes,
            )
        except RecordNotFoundError as error:
            raise HTTPException(
                status_code=404,
                detail=f"Training review item not found: {error.args[0]}",
            ) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @application.post("/api/utterances", response_model=UtteranceRecord, status_code=201)
    async def create_utterance(
        file: Annotated[UploadFile, File(description="Original user audio")],
        speaker_id: Annotated[str, Form(min_length=1)],
        surface_text: Annotated[str, Form(min_length=1)],
        normalized_meaning: Annotated[str | None, Form()] = None,
        region: Annotated[str, Form(min_length=1)] = "Busan",
        device: Annotated[str, Form(min_length=1)] = "unknown",
        environment: Annotated[str, Form(min_length=1)] = "unknown",
        dialect_expressions: Annotated[str, Form()] = "[]",
        storage_allowed: Annotated[bool, Form()] = False,
        research_use_allowed: Annotated[bool, Form()] = False,
        model_training_allowed: Annotated[bool, Form()] = False,
    ) -> UtteranceRecord:
        if not storage_allowed:
            raise HTTPException(
                status_code=422,
                detail="Persistent lab upload requires explicit audio storage consent.",
            )
        staged_path = storage.staging_dir / f"{uuid4()}.upload"
        total_bytes = 0
        try:
            with staged_path.open("wb") as stream:
                while chunk := await file.read(1024 * 1024):
                    total_bytes += len(chunk)
                    if total_bytes > settings.max_upload_bytes:
                        raise HTTPException(status_code=413, detail="Audio upload is too large.")
                    stream.write(chunk)

            try:
                raw_labels = json.loads(dialect_expressions)
                if not isinstance(raw_labels, list):
                    raise TypeError("dialect_expressions must be a JSON array")
                labels = expand_dialect_expression_labels(
                    tuple(DialectExpressionLabel.model_validate(item) for item in raw_labels)
                )
            except (json.JSONDecodeError, TypeError, ValidationError, ValueError) as error:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid dialect expression labels: {error}",
                ) from error

            audio = processor.process(staged_path, file.filename or "upload.audio")
            record = UtteranceRecord(
                speaker=SpeakerContext(
                    speaker_id=speaker_id,
                    region=region,
                    device=device,
                    environment=environment,
                ),
                consent=ConsentRecord(
                    storage_allowed=storage_allowed,
                    research_use_allowed=research_use_allowed,
                    model_training_allowed=model_training_allowed,
                ),
                audio=audio,
                ground_truth=LinguisticGroundTruth(
                    surface_text=surface_text,
                    normalized_meaning=normalized_meaning or None,
                    dialect_expressions=labels,
                ),
            )
            storage.save_utterance(record)
            return record
        except AudioValidationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        finally:
            staged_path.unlink(missing_ok=True)
            await file.close()

    @application.get("/api/utterances/{utterance_id}", response_model=UtteranceRecord)
    async def get_utterance(utterance_id: UUID) -> UtteranceRecord:
        try:
            return storage.load_utterance(utterance_id)
        except RecordNotFoundError as error:
            raise HTTPException(status_code=404, detail="Utterance not found.") from error

    @application.patch(
        "/api/utterances/{utterance_id}/labels",
        response_model=UtteranceRecord,
    )
    async def update_utterance_labels(
        utterance_id: UUID,
        request: UpdateLabelRequest,
    ) -> UtteranceRecord:
        try:
            record = storage.load_utterance(utterance_id)
        except RecordNotFoundError as error:
            raise HTTPException(status_code=404, detail="Utterance not found.") from error
        if storage.utterance_is_frozen(utterance_id):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Labels in a frozen benchmark cannot be changed. "
                    "Create a new benchmark version or correct the label before freezing."
                ),
            )

        try:
            labels = expand_dialect_expression_labels(request.dialect_expressions)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        revision_number = len(storage.list_label_revisions(utterance_id)) + 1
        updated_ground_truth = LinguisticGroundTruth(
            surface_text=request.surface_text,
            normalized_meaning=request.normalized_meaning or None,
            dialect_expressions=labels,
            label_status=record.ground_truth.label_status,
            label_version=f"label_v{revision_number}",
            reviewer_id=request.changed_by,
        )
        if updated_ground_truth.model_dump(
            exclude={"label_version", "reviewer_id"}
        ) == record.ground_truth.model_dump(exclude={"label_version", "reviewer_id"}):
            raise HTTPException(status_code=409, detail="The submitted label has no changes.")

        revision = LabelRevision(
            utterance_id=utterance_id,
            previous=record.ground_truth,
            updated=updated_ground_truth,
            changed_by=request.changed_by,
            reason=request.reason,
        )
        updated_record = record.model_copy(update={"ground_truth": updated_ground_truth})
        storage.save_label_revision(revision)
        storage.save_utterance(updated_record)
        return updated_record

    @application.get(
        "/api/utterances/{utterance_id}/label-revisions",
        response_model=list[LabelRevision],
    )
    async def list_label_revisions(utterance_id: UUID) -> list[LabelRevision]:
        try:
            storage.load_utterance(utterance_id)
        except RecordNotFoundError as error:
            raise HTTPException(status_code=404, detail="Utterance not found.") from error
        return list(storage.list_label_revisions(utterance_id))

    @application.delete(
        "/api/utterances/{utterance_id}",
        response_model=DeleteUtteranceResponse,
    )
    async def delete_utterance(utterance_id: UUID) -> DeleteUtteranceResponse:
        try:
            record = storage.load_utterance(utterance_id)
        except RecordNotFoundError as error:
            raise HTTPException(status_code=404, detail="Utterance not found.") from error
        if storage.utterance_is_frozen(utterance_id):
            raise HTTPException(
                status_code=409,
                detail="An utterance in a frozen benchmark cannot be deleted.",
            )
        if storage.utterance_is_in_training_import(utterance_id):
            raise HTTPException(
                status_code=409,
                detail=(
                    "An utterance referenced by a TASK-004 training import cannot "
                    "be deleted. Mark it for re-recording instead."
                ),
            )
        archived = storage.archive_utterance(record)
        return DeleteUtteranceResponse(
            utterance_id=utterance_id,
            archive_id=archived.archive_id,
            archived_paths=archived.archived_paths,
            preserved_shared_paths=archived.preserved_shared_paths,
            removed_export_rows=archived.removed_export_rows,
        )

    @application.get("/api/utterances/{utterance_id}/audio/{role}")
    async def get_audio(utterance_id: UUID, role: AudioRole) -> FileResponse:
        try:
            record = storage.load_utterance(utterance_id)
        except RecordNotFoundError as error:
            raise HTTPException(status_code=404, detail="Utterance not found.") from error
        try:
            asset = record.audio.asset_for_role(role)
        except KeyError as error:
            raise HTTPException(
                status_code=404,
                detail=f"Audio role is not available for this record: {role.value}",
            ) from error
        path = storage.resolve(asset.relative_path)
        if not path.is_file():
            raise HTTPException(status_code=410, detail="Audio asset is missing.")
        if file_is_dataless(path):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Audio asset is stored in cloud-only mode. "
                    "Download data/lab to this Mac and retry."
                ),
            )
        return FileResponse(
            path,
            media_type="audio/wav" if path.suffix.lower() == ".wav" else "application/octet-stream",
            filename=path.name,
            content_disposition_type="inline",
        )

    @application.get("/api/utterances/{utterance_id}/analysis")
    def get_analysis(utterance_id: UUID) -> object:
        """Run blocking audio reads and analysis in FastAPI's worker thread."""

        try:
            record = storage.load_utterance(utterance_id)
        except RecordNotFoundError as error:
            raise HTTPException(status_code=404, detail="Utterance not found.") from error
        path = storage.resolve(record.audio.derived.relative_path)
        if file_is_dataless(path):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Audio asset is stored in cloud-only mode. "
                    "Download data/lab to this Mac and retry."
                ),
            )
        return analyze_audio(path)

    @application.post("/api/evaluations", response_model=EvaluationCaseResult)
    async def create_evaluation(request: ManualEvaluationRequest) -> EvaluationCaseResult:
        try:
            prediction = evaluate_and_store(
                request=request,
                experiment_id=f"manual-{uuid4()}",
                source=PredictionSource.MANUAL,
            )
        except RecordNotFoundError as error:
            raise HTTPException(status_code=404, detail="Utterance not found.") from error
        return prediction.evaluation

    @application.get("/api/experiments", response_model=list[ExperimentRun])
    async def list_experiments() -> list[ExperimentRun]:
        return list(storage.list_experiments())

    @application.post("/api/experiments", response_model=ExperimentRun, status_code=201)
    async def create_experiment(experiment: ExperimentRun) -> ExperimentRun:
        try:
            storage.save_experiment(experiment)
            return storage.load_experiment(experiment.experiment_id)
        except FileExistsError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @application.get("/api/predictions", response_model=list[StoredPrediction])
    async def list_predictions(utterance_id: UUID | None = None) -> list[StoredPrediction]:
        return list(storage.list_predictions(utterance_id))

    @application.get("/api/predictions/{prediction_id}", response_model=StoredPrediction)
    async def get_prediction(prediction_id: UUID) -> StoredPrediction:
        try:
            return storage.load_prediction(prediction_id)
        except RecordNotFoundError as error:
            raise HTTPException(status_code=404, detail="Prediction not found.") from error

    @application.get(
        "/api/predictions/{prediction_id}/diagnostics",
        response_model=PredictionDiagnostics,
    )
    async def get_prediction_diagnostics(prediction_id: UUID) -> PredictionDiagnostics:
        try:
            prediction = storage.load_prediction(prediction_id)
            return diagnose_prediction(storage, prediction)
        except RecordNotFoundError as error:
            raise HTTPException(status_code=404, detail="Prediction not found.") from error

    @application.post("/api/predictions", response_model=StoredPrediction, status_code=201)
    async def create_prediction(request: CreatePredictionRequest) -> StoredPrediction:
        try:
            return evaluate_and_store(
                request=request,
                experiment_id=request.experiment_id,
                source=PredictionSource.MANUAL,
                latency_ms=request.latency_ms,
            )
        except RecordNotFoundError as error:
            raise HTTPException(status_code=404, detail="Utterance not found.") from error
        except FileExistsError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @application.get("/api/reviews", response_model=list[HumanReview])
    async def list_reviews(prediction_id: UUID | None = None) -> list[HumanReview]:
        return list(storage.list_reviews(prediction_id))

    @application.get("/api/review-queue", response_model=list[StoredPrediction])
    async def review_queue() -> list[StoredPrediction]:
        reviewed_prediction_ids = {
            review.prediction_id for review in storage.list_reviews()
        }
        return [
            prediction
            for prediction in storage.list_predictions()
            if prediction.automatic_failure_candidates
            and prediction.prediction_id not in reviewed_prediction_ids
        ]

    @application.post("/api/reviews", response_model=HumanReview, status_code=201)
    async def create_review(request: CreateHumanReviewRequest) -> HumanReview:
        try:
            prediction = storage.load_prediction(request.prediction_id)
        except RecordNotFoundError as error:
            raise HTTPException(status_code=404, detail="Prediction not found.") from error
        review = HumanReview(
            prediction_id=prediction.prediction_id,
            utterance_id=prediction.utterance_id,
            reviewer_id=request.reviewer_id,
            verdict=request.verdict,
            confirmed_failure_types=request.confirmed_failure_types,
            notes=request.notes,
        )
        storage.save_review(review)
        return review

    @application.post("/api/comparisons", response_model=PredictionComparison)
    async def compare_predictions(request: ComparePredictionsRequest) -> PredictionComparison:
        if request.prediction_a_id == request.prediction_b_id:
            raise HTTPException(status_code=409, detail="Select two different predictions.")
        try:
            prediction_a = storage.load_prediction(request.prediction_a_id)
            prediction_b = storage.load_prediction(request.prediction_b_id)
        except RecordNotFoundError as error:
            raise HTTPException(status_code=404, detail="Prediction not found.") from error
        if prediction_a.utterance_id != prediction_b.utterance_id:
            raise HTTPException(
                status_code=409,
                detail="A/B comparison requires predictions for the same utterance.",
            )
        evaluation_a = prediction_a.evaluation
        evaluation_b = prediction_b.evaluation
        return PredictionComparison(
            utterance_id=prediction_a.utterance_id,
            prediction_a=prediction_a,
            prediction_b=prediction_b,
            cer_delta_b_minus_a=evaluation_b.cer - evaluation_a.cer,
            preservation_delta_b_minus_a=(
                evaluation_b.dialect.preservation_rate
                - evaluation_a.dialect.preservation_rate
            ),
            overcorrection_delta_b_minus_a=(
                evaluation_b.dialect.overcorrection_rate
                - evaluation_a.dialect.overcorrection_rate
            ),
            confidence_delta_b_minus_a=(
                evaluation_b.confidence - evaluation_a.confidence
                if evaluation_a.confidence is not None
                and evaluation_b.confidence is not None
                else None
            ),
        )

    @application.get("/api/benchmarks", response_model=list[BenchmarkManifest])
    async def list_benchmarks() -> list[BenchmarkManifest]:
        return list(storage.list_manifests())

    @application.post("/api/benchmarks", response_model=BenchmarkManifest, status_code=201)
    async def create_benchmark(request: CreateBenchmarkRequest) -> BenchmarkManifest:
        try:
            records = [storage.load_utterance(value) for value in request.utterance_ids]
            manifest = build_manifest(
                benchmark_id=request.benchmark_id,
                benchmark_version=request.benchmark_version,
                records=records,
                split=request.split,
            )
            storage.save_manifest(manifest)
            return manifest
        except RecordNotFoundError as error:
            raise HTTPException(
                status_code=404, detail=f"Utterance not found: {error.args[0]}"
            ) from error
        except (ValueError, FileExistsError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @application.delete(
        "/api/benchmarks/{benchmark_id}/{benchmark_version}",
        response_model=DeleteBenchmarkResponse,
    )
    async def delete_benchmark(
        benchmark_id: str,
        benchmark_version: str,
    ) -> DeleteBenchmarkResponse:
        try:
            manifest = storage.load_manifest(benchmark_id, benchmark_version)
            archived = storage.archive_manifest(benchmark_id, benchmark_version)
        except RecordNotFoundError as error:
            raise HTTPException(status_code=404, detail="Benchmark not found.") from error
        return DeleteBenchmarkResponse(
            benchmark_id=manifest.benchmark_id,
            benchmark_version=manifest.benchmark_version,
            archive_id=archived.archive_id,
            archived_path=archived.archived_path,
            entry_count=len(manifest.entries),
        )

    @application.get("/api/schemas/{schema_name}")
    async def get_schema(schema_name: str) -> dict[str, object]:
        schemas: dict[str, type[StrictSchema]] = {
            "utterance": UtteranceRecord,
            "benchmark": BenchmarkManifest,
            "evaluation": EvaluationCaseResult,
            "experiment": ExperimentRun,
            "prediction": StoredPrediction,
            "review": HumanReview,
            "comparison": PredictionComparison,
            "label-revision": LabelRevision,
            "training-dataset": TrainingDatasetManifest,
            "training-recording-import-manifest": TrainingRecordingImportManifest,
            "training-recording-import-plan": TrainingRecordingImportPlan,
            "training-recording-review-queue": TrainingRecordingReviewQueue,
            "training-recording-review-request": TrainingRecordingReviewRequest,
        }
        schema = schemas.get(schema_name)
        if schema is None:
            raise HTTPException(status_code=404, detail="Unknown schema.")
        return cast(dict[str, object], schema.model_json_schema())

    return application


app = create_app()
