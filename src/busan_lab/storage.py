"""Deterministic local storage for immutable assets and versioned records."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel

from busan_lab.schemas.benchmark import BenchmarkManifest
from busan_lab.schemas.evaluation import ErrorExportRecord
from busan_lab.schemas.experiment import ExperimentRun, HumanReview, StoredPrediction
from busan_lab.schemas.training import TrainingDatasetManifest
from busan_lab.schemas.training_import import TrainingRecordingImportManifest
from busan_lab.schemas.utterance import LabelRevision, UtteranceRecord

MACOS_DATALESS_FLAG = 0x40000000


class RecordNotFoundError(KeyError):
    """Raised when an immutable lab record does not exist."""


class DatalessFileError(OSError):
    """Raised instead of blocking forever on a cloud-evicted record file."""

    def __init__(self, path: Path) -> None:
        super().__init__(
            f"Record file is stored in cloud-only mode: {path}. "
            "Download data/lab to this Mac (e.g. `brctl download <file>`) and retry."
        )
        self.path = path


def _read_text(path: Path) -> str:
    """Read a record file, failing fast if macOS evicted its contents."""

    if file_is_dataless(path):
        raise DatalessFileError(path)
    return path.read_text(encoding="utf-8")


@dataclass(frozen=True)
class ArchivedUtterance:
    """Recoverable archive result for one utterance and its dependent evidence."""

    archive_id: str
    archived_paths: tuple[str, ...]
    preserved_shared_paths: tuple[str, ...]
    removed_export_rows: int


@dataclass(frozen=True)
class ArchivedBenchmark:
    """Recoverable archive result for one frozen benchmark manifest."""

    archive_id: str
    archived_path: str


class LabStorage:
    """Filesystem repository with atomic JSON record writes."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.raw_dir = self.root / "raw"
        self.master_dir = self.root / "master"
        self.derived_dir = self.root / "derived"
        self.staging_dir = self.root / "staging"
        self.records_dir = self.root / "records"
        self.label_revisions_dir = self.root / "label_revisions"
        self.manifests_dir = self.root / "manifests"
        self.training_datasets_dir = self.root / "training_datasets"
        self.training_imports_dir = self.root / "training_imports"
        self.exports_dir = self.root / "exports"
        self.reports_dir = self.root / "reports"
        self.experiments_dir = self.root / "experiments"
        self.predictions_dir = self.root / "predictions"
        self.reviews_dir = self.root / "reviews"
        self.trash_dir = self.root / "trash"
        for directory in (
            self.raw_dir,
            self.master_dir,
            self.derived_dir,
            self.staging_dir,
            self.records_dir,
            self.label_revisions_dir,
            self.manifests_dir,
            self.training_datasets_dir,
            self.training_imports_dir,
            self.exports_dir,
            self.reports_dir,
            self.experiments_dir,
            self.predictions_dir,
            self.reviews_dir,
            self.trash_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def relative(self, path: Path) -> str:
        """Return a POSIX path relative to the storage root."""

        return path.resolve().relative_to(self.root).as_posix()

    def resolve(self, relative_path: str) -> Path:
        """Resolve a stored relative path without allowing path traversal."""

        candidate = (self.root / relative_path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError("path escapes the configured data root")
        return candidate

    def save_utterance(self, record: UtteranceRecord) -> Path:
        path = self.records_dir / f"{record.utterance_id}.json"
        self._atomic_model_write(path, record)
        return path

    def load_utterance(self, utterance_id: UUID | str) -> UtteranceRecord:
        path = self.records_dir / f"{utterance_id}.json"
        if not path.is_file():
            raise RecordNotFoundError(str(utterance_id))
        return UtteranceRecord.model_validate_json(_read_text(path))

    def list_utterances(self) -> tuple[UtteranceRecord, ...]:
        return tuple(
            UtteranceRecord.model_validate_json(_read_text(path))
            for path in sorted(self.records_dir.glob("*.json"))
        )

    def save_label_revision(self, revision: LabelRevision) -> Path:
        directory = self.label_revisions_dir / str(revision.utterance_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{revision.revision_id}.json"
        if path.exists():
            raise FileExistsError(f"label revision already exists: {revision.revision_id}")
        self._atomic_model_write(path, revision)
        return path

    def list_label_revisions(
        self,
        utterance_id: UUID | str,
    ) -> tuple[LabelRevision, ...]:
        directory = self.label_revisions_dir / str(utterance_id)
        if not directory.is_dir():
            return ()
        return tuple(
            LabelRevision.model_validate_json(_read_text(path))
            for path in sorted(directory.glob("*.json"))
        )

    def utterance_is_frozen(self, utterance_id: UUID) -> bool:
        frozen_in_benchmark = any(
            manifest.frozen
            and any(entry.utterance_id == utterance_id for entry in manifest.entries)
            for manifest in self.list_manifests()
        )
        frozen_in_training = any(
            manifest.frozen
            and any(entry.utterance_id == utterance_id for entry in manifest.entries)
            for manifest in self.list_training_datasets()
        )
        return frozen_in_benchmark or frozen_in_training

    def utterance_is_in_training_import(self, utterance_id: UUID) -> bool:
        return any(
            any(entry.utterance_id == utterance_id for entry in manifest.entries)
            for manifest in self.list_training_recording_imports()
        )

    def archive_utterance(self, record: UtteranceRecord) -> ArchivedUtterance:
        """Remove an utterance from the active lab while keeping a recoverable archive."""

        archive_id = str(uuid4())
        archive_root = self.trash_dir / str(record.utterance_id) / archive_id
        archived_paths: list[str] = []

        other_asset_paths = {
            asset.relative_path
            for other_record in self.list_utterances()
            if other_record.utterance_id != record.utterance_id
            for asset in other_record.audio.lineage_assets()
        }
        preserved_shared_paths: list[str] = []
        targets: set[Path] = {
            self.records_dir / f"{record.utterance_id}.json",
        }
        for asset in record.audio.lineage_assets():
            if asset.relative_path in other_asset_paths:
                preserved_shared_paths.append(asset.relative_path)
            else:
                targets.add(self.resolve(asset.relative_path))

        revision_directory = self.label_revisions_dir / str(record.utterance_id)
        targets.update(revision_directory.glob("*.json"))

        predictions = self.list_predictions(record.utterance_id)
        prediction_ids = {prediction.prediction_id for prediction in predictions}
        targets.update(
            self.predictions_dir / f"{prediction.prediction_id}.json"
            for prediction in predictions
        )
        targets.update(
            self.reviews_dir / f"{review.review_id}.json"
            for review in self.list_reviews()
            if review.prediction_id in prediction_ids
        )

        utterance_text = str(record.utterance_id)
        for report_path in self.reports_dir.glob("*.json"):
            if utterance_text in _read_text(report_path):
                targets.add(report_path)

        for target in sorted(targets):
            if target.is_file():
                archived_paths.append(self._move_to_archive(target, archive_root))

        removed_export_rows = self._remove_utterance_from_error_export(
            record.utterance_id,
            archive_root,
            archived_paths,
        )
        return ArchivedUtterance(
            archive_id=archive_id,
            archived_paths=tuple(archived_paths),
            preserved_shared_paths=tuple(sorted(set(preserved_shared_paths))),
            removed_export_rows=removed_export_rows,
        )

    def save_manifest(self, manifest: BenchmarkManifest) -> Path:
        safe_name = _safe_record_name(f"{manifest.benchmark_id}--{manifest.benchmark_version}")
        path = self.manifests_dir / f"{safe_name}.json"
        if path.exists() and manifest.frozen:
            existing = BenchmarkManifest.model_validate_json(_read_text(path))
            if existing != manifest:
                raise FileExistsError(
                    "a frozen benchmark version already exists with different content"
                )
            return path
        self._atomic_model_write(path, manifest)
        return path

    def load_manifest(self, benchmark_id: str, benchmark_version: str) -> BenchmarkManifest:
        safe_name = _safe_record_name(f"{benchmark_id}--{benchmark_version}")
        path = self.manifests_dir / f"{safe_name}.json"
        if not path.is_file():
            raise RecordNotFoundError(f"{benchmark_id}@{benchmark_version}")
        return BenchmarkManifest.model_validate_json(_read_text(path))

    def list_manifests(self) -> tuple[BenchmarkManifest, ...]:
        return tuple(
            BenchmarkManifest.model_validate_json(_read_text(path))
            for path in sorted(self.manifests_dir.glob("*.json"))
        )

    def save_training_dataset(self, manifest: TrainingDatasetManifest) -> Path:
        safe_name = _safe_record_name(
            f"{manifest.dataset_id}--{manifest.dataset_version}"
        )
        path = self.training_datasets_dir / f"{safe_name}.json"
        if path.exists():
            existing = TrainingDatasetManifest.model_validate_json(
                _read_text(path)
            )
            if existing != manifest:
                raise FileExistsError(
                    "a frozen training dataset version already exists with "
                    "different content"
                )
            return path
        self._atomic_model_write(path, manifest)
        return path

    def load_training_dataset(
        self,
        dataset_id: str,
        dataset_version: str,
    ) -> TrainingDatasetManifest:
        safe_name = _safe_record_name(f"{dataset_id}--{dataset_version}")
        path = self.training_datasets_dir / f"{safe_name}.json"
        if not path.is_file():
            raise RecordNotFoundError(f"{dataset_id}@{dataset_version}")
        return TrainingDatasetManifest.model_validate_json(
            _read_text(path)
        )

    def list_training_datasets(self) -> tuple[TrainingDatasetManifest, ...]:
        return tuple(
            TrainingDatasetManifest.model_validate_json(
                _read_text(path)
            )
            for path in sorted(self.training_datasets_dir.glob("*.json"))
        )

    def training_recording_import_exists(self, import_id: str) -> bool:
        safe_name = _safe_record_name(import_id)
        return (self.training_imports_dir / f"{safe_name}.json").is_file()

    def save_training_recording_import(
        self,
        manifest: TrainingRecordingImportManifest,
    ) -> Path:
        safe_name = _safe_record_name(manifest.import_id)
        path = self.training_imports_dir / f"{safe_name}.json"
        if path.exists():
            existing = TrainingRecordingImportManifest.model_validate_json(
                _read_text(path)
            )
            if existing != manifest:
                raise FileExistsError(
                    "training recording import already exists with different content"
                )
            return path
        self._atomic_model_write(path, manifest)
        return path

    def load_training_recording_import(
        self,
        import_id: str,
    ) -> TrainingRecordingImportManifest:
        safe_name = _safe_record_name(import_id)
        path = self.training_imports_dir / f"{safe_name}.json"
        if not path.is_file():
            raise RecordNotFoundError(import_id)
        return TrainingRecordingImportManifest.model_validate_json(
            _read_text(path)
        )

    def list_training_recording_imports(
        self,
    ) -> tuple[TrainingRecordingImportManifest, ...]:
        return tuple(
            TrainingRecordingImportManifest.model_validate_json(
                _read_text(path)
            )
            for path in sorted(self.training_imports_dir.glob("*.json"))
        )

    def archive_manifest(self, benchmark_id: str, benchmark_version: str) -> ArchivedBenchmark:
        safe_name = _safe_record_name(f"{benchmark_id}--{benchmark_version}")
        path = self.manifests_dir / f"{safe_name}.json"
        if not path.is_file():
            raise RecordNotFoundError(f"{benchmark_id}@{benchmark_version}")
        archive_id = str(uuid4())
        destination = self.trash_dir / "benchmarks" / safe_name / archive_id / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, destination)
        return ArchivedBenchmark(
            archive_id=archive_id,
            archived_path=self.relative(destination),
        )

    def append_error(self, record: ErrorExportRecord) -> Path:
        path = self.exports_dir / "normalization-errors.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(record.model_dump_json())
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return path

    def save_experiment(self, experiment: ExperimentRun) -> Path:
        safe_name = _safe_record_name(experiment.experiment_id)
        path = self.experiments_dir / f"{safe_name}.json"
        if path.exists():
            existing = ExperimentRun.model_validate_json(_read_text(path))
            existing_conditions = (
                existing.task,
                existing.model,
                existing.benchmark_id,
                existing.benchmark_version,
                existing.hypothesis,
                existing.changed_variable,
            )
            requested_conditions = (
                experiment.task,
                experiment.model,
                experiment.benchmark_id,
                experiment.benchmark_version,
                experiment.hypothesis,
                experiment.changed_variable,
            )
            if existing_conditions != requested_conditions:
                raise FileExistsError(
                    "experiment_id already exists with different model or conditions"
                )
            return path
        self._atomic_model_write(path, experiment)
        return path

    def load_experiment(self, experiment_id: str) -> ExperimentRun:
        safe_name = _safe_record_name(experiment_id)
        path = self.experiments_dir / f"{safe_name}.json"
        if not path.is_file():
            raise RecordNotFoundError(experiment_id)
        return ExperimentRun.model_validate_json(_read_text(path))

    def list_experiments(self) -> tuple[ExperimentRun, ...]:
        return tuple(
            ExperimentRun.model_validate_json(_read_text(path))
            for path in sorted(self.experiments_dir.glob("*.json"))
        )

    def save_prediction(self, prediction: StoredPrediction) -> Path:
        path = self.predictions_dir / f"{prediction.prediction_id}.json"
        if path.exists():
            raise FileExistsError(f"prediction already exists: {prediction.prediction_id}")
        self._atomic_model_write(path, prediction)
        return path

    def load_prediction(self, prediction_id: UUID | str) -> StoredPrediction:
        path = self.predictions_dir / f"{prediction_id}.json"
        if not path.is_file():
            raise RecordNotFoundError(str(prediction_id))
        return StoredPrediction.model_validate_json(_read_text(path))

    def list_predictions(
        self,
        utterance_id: UUID | None = None,
    ) -> tuple[StoredPrediction, ...]:
        predictions = tuple(
            StoredPrediction.model_validate_json(_read_text(path))
            for path in sorted(self.predictions_dir.glob("*.json"))
        )
        if utterance_id is None:
            return predictions
        return tuple(
            prediction
            for prediction in predictions
            if prediction.utterance_id == utterance_id
        )

    def save_review(self, review: HumanReview) -> Path:
        path = self.reviews_dir / f"{review.review_id}.json"
        if path.exists():
            raise FileExistsError(f"review already exists: {review.review_id}")
        self._atomic_model_write(path, review)
        return path

    def list_reviews(self, prediction_id: UUID | None = None) -> tuple[HumanReview, ...]:
        reviews = tuple(
            HumanReview.model_validate_json(_read_text(path))
            for path in sorted(self.reviews_dir.glob("*.json"))
        )
        if prediction_id is None:
            return reviews
        return tuple(review for review in reviews if review.prediction_id == prediction_id)

    def _atomic_model_write(self, path: Path, model: BaseModel) -> None:
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(
            model.model_dump_json(indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _move_to_archive(self, path: Path, archive_root: Path) -> str:
        relative_path = self.relative(path)
        destination = archive_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, destination)
        return relative_path

    def _remove_utterance_from_error_export(
        self,
        utterance_id: UUID,
        archive_root: Path,
        archived_paths: list[str],
    ) -> int:
        path = self.exports_dir / "normalization-errors.jsonl"
        if not path.is_file():
            return 0
        kept_lines: list[str] = []
        removed_rows = 0
        for line in _read_text(path).splitlines():
            try:
                payload = json.loads(line)
                exported_utterance_id = payload.get("case", {}).get("utterance_id")
            except (AttributeError, json.JSONDecodeError):
                exported_utterance_id = None
            if exported_utterance_id == str(utterance_id):
                removed_rows += 1
            else:
                kept_lines.append(line)
        if not removed_rows:
            return 0
        archived_paths.append(self._move_to_archive(path, archive_root))
        path.write_text(
            "\n".join(kept_lines) + ("\n" if kept_lines else ""),
            encoding="utf-8",
        )
        return removed_rows


def file_is_dataless(path: Path) -> bool:
    """Return whether macOS has evicted a file's contents to cloud storage."""

    flags = getattr(path.stat(), "st_flags", 0)
    return bool(flags & MACOS_DATALESS_FLAG)


def write_jsonl(path: Path, models: Iterable[BaseModel]) -> None:
    """Write deterministic JSONL through a temporary sibling file."""

    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for model in models:
            stream.write(model.model_dump_json())
            stream.write("\n")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(_read_text(path))


def _safe_record_name(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "-_." else "-" for character in value
    )
    if not safe.strip(".-"):
        raise ValueError("record name contains no safe characters")
    return safe[:180]
