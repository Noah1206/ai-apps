from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from busan_lab.audio import AudioProcessor
from busan_lab.cli import export_schemas, main
from busan_lab.config import LabSettings
from busan_lab.manifest import build_manifest
from busan_lab.schemas.common import (
    ConsentRecord,
    DatasetSplit,
    ReviewStatus,
)
from busan_lab.schemas.training import TrainingSplitAssignments
from busan_lab.schemas.utterance import (
    LinguisticGroundTruth,
    SpeakerContext,
    UtteranceRecord,
)
from busan_lab.storage import LabStorage
from busan_lab.training import (
    TrainingDatasetValidationError,
    build_training_dataset,
    export_training_dataset_bundle,
    review_training_label,
)
from tests.helpers import make_wav_bytes


def make_record(
    root: Path,
    *,
    index: int,
    speaker_id: str,
    surface_text: str,
    frequency_hz: float,
    training_allowed: bool = True,
    label_status: ReviewStatus = ReviewStatus.APPROVED,
) -> UtteranceRecord:
    storage = LabStorage(root / "lab")
    source = root / f"source-{index}.wav"
    source.write_bytes(make_wav_bytes(frequency_hz=frequency_hz))
    audio = AudioProcessor(
        LabSettings.from_environment(storage.root),
        storage,
    ).process(source, source.name)
    return UtteranceRecord(
        speaker=SpeakerContext(
            speaker_id=speaker_id,
            region="Busan",
            device="fixture",
            environment="quiet",
        ),
        consent=ConsentRecord(
            storage_allowed=True,
            research_use_allowed=True,
            model_training_allowed=training_allowed,
        ),
        audio=audio,
        ground_truth=LinguisticGroundTruth(
            surface_text=surface_text,
            normalized_meaning="학습 target으로 사용하지 않는 의미",
            label_status=label_status,
            label_version="label_v1",
            reviewer_id="reviewer-fixture",
        ),
    )


def frozen_benchmark(root: Path):
    record = make_record(
        root,
        index=900,
        speaker_id="benchmark-speaker",
        surface_text="국밥 하나 주이소",
        frequency_hz=180,
    )
    return build_manifest(
        benchmark_id="busan-surface-v0",
        benchmark_version="1.0.0",
        records=[record],
        split=DatasetSplit.TEST,
    )


def valid_training_fixture(root: Path):
    train = make_record(
        root,
        index=1,
        speaker_id="train-speaker",
        surface_text="마 오늘 날씨 좋노",
        frequency_hz=220,
    )
    validation = make_record(
        root,
        index=2,
        speaker_id="validation-speaker",
        surface_text="니 지금 어데고",
        frequency_hz=330,
    )
    assignments = TrainingSplitAssignments(
        train_utterance_ids=(train.utterance_id,),
        validation_utterance_ids=(validation.utterance_id,),
    )
    return train, validation, assignments


def test_valid_training_manifest_and_model_neutral_export(tmp_path: Path) -> None:
    train, validation, assignments = valid_training_fixture(tmp_path)
    benchmark = frozen_benchmark(tmp_path)
    manifest, report = build_training_dataset(
        dataset_id="busan-asr-training-pilot-v0",
        dataset_version="0.1.0",
        records=(train, validation),
        assignments=assignments,
        benchmark_manifests=(benchmark,),
    )

    assert report.passed is True
    assert report.train_utterance_count == 1
    assert report.validation_utterance_count == 1
    assert any("300-500" in warning for warning in report.warnings)
    assert {entry.split for entry in manifest.entries} == {
        DatasetSplit.TRAIN,
        DatasetSplit.VALIDATION,
    }
    assert all(entry.model_training_allowed for entry in manifest.entries)
    assert all(entry.audio_quality_passed for entry in manifest.entries)

    storage = LabStorage(tmp_path / "lab")
    storage.save_utterance(train)
    storage.save_utterance(validation)
    stored_path = storage.save_training_dataset(manifest)
    assert stored_path.is_file()
    assert storage.load_training_dataset(
        manifest.dataset_id,
        manifest.dataset_version,
    ) == manifest
    assert storage.utterance_is_frozen(train.utterance_id) is True
    assert storage.save_training_dataset(manifest) == stored_path
    with pytest.raises(FileExistsError, match="different content"):
        storage.save_training_dataset(
            manifest.model_copy(update={"entries": tuple(reversed(manifest.entries))})
        )

    output = export_training_dataset_bundle(
        storage,
        manifest,
        report,
        tmp_path / "training-export.zip",
    )
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert {
            "training_dataset.json",
            "validation_report.json",
            "schemas/training-dataset.schema.json",
            "schemas/training-export-record.schema.json",
            "manifests/train.jsonl",
            "manifests/validation.jsonl",
            train.audio.derived.relative_path,
            validation.audio.derived.relative_path,
        } == names
        train_row = json.loads(archive.read("manifests/train.jsonl"))
        validation_row = json.loads(archive.read("manifests/validation.jsonl"))
        assert train_row["text"] == train.ground_truth.surface_text
        assert validation_row["text"] == validation.ground_truth.surface_text
        assert "normalized_meaning" not in train_row
        assert train_row["split"] == "train"
        assert validation_row["split"] == "validation"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export_training_dataset_bundle(storage, manifest, report, output)


def test_training_requires_explicit_training_consent(tmp_path: Path) -> None:
    train, validation, assignments = valid_training_fixture(tmp_path)
    train = train.model_copy(
        update={
            "consent": train.consent.model_copy(
                update={"model_training_allowed": False}
            )
        }
    )

    with pytest.raises(ValueError, match="no explicit model-training consent"):
        build_training_dataset(
            dataset_id="consent-test",
            dataset_version="0.1.0",
            records=(train, validation),
            assignments=assignments,
            benchmark_manifests=(frozen_benchmark(tmp_path),),
        )


def test_training_requires_human_reviewed_surface_label(tmp_path: Path) -> None:
    train, validation, assignments = valid_training_fixture(tmp_path)
    train = train.model_copy(
        update={
            "ground_truth": train.ground_truth.model_copy(
                update={"label_status": ReviewStatus.CANDIDATE}
            )
        }
    )

    with pytest.raises(ValueError, match="human review is required"):
        build_training_dataset(
            dataset_id="label-test",
            dataset_version="0.1.0",
            records=(train, validation),
            assignments=assignments,
            benchmark_manifests=(frozen_benchmark(tmp_path),),
        )


def test_training_manifest_rejects_speaker_crossing_splits(tmp_path: Path) -> None:
    train, validation, assignments = valid_training_fixture(tmp_path)
    validation = validation.model_copy(
        update={
            "speaker": validation.speaker.model_copy(
                update={"speaker_id": train.speaker.speaker_id}
            )
        }
    )

    with pytest.raises(ValidationError, match="speaker"):
        build_training_dataset(
            dataset_id="speaker-leak-test",
            dataset_version="0.1.0",
            records=(train, validation),
            assignments=assignments,
            benchmark_manifests=(frozen_benchmark(tmp_path),),
        )


def test_training_manifest_rejects_duplicate_audio_lineage(tmp_path: Path) -> None:
    train, validation, assignments = valid_training_fixture(tmp_path)
    validation = validation.model_copy(update={"audio": train.audio})

    with pytest.raises(ValidationError, match="duplicate audio lineage"):
        build_training_dataset(
            dataset_id="audio-duplicate-test",
            dataset_version="0.1.0",
            records=(train, validation),
            assignments=assignments,
            benchmark_manifests=(frozen_benchmark(tmp_path),),
        )


@pytest.mark.parametrize(
    ("leak_kind", "message"),
    [
        ("utterance", "utterance"),
        ("speaker", "speaker"),
        ("audio", "audio lineage"),
        ("surface", "exact Surface transcript"),
    ],
)
def test_frozen_benchmark_leakage_is_rejected(
    tmp_path: Path,
    leak_kind: str,
    message: str,
) -> None:
    benchmark = frozen_benchmark(tmp_path)
    benchmark_record = make_record(
        tmp_path,
        index=901,
        speaker_id="benchmark-speaker",
        surface_text="국밥 하나 주이소",
        frequency_hz=180,
    )
    train, validation, assignments = valid_training_fixture(tmp_path)
    if leak_kind == "utterance":
        train = train.model_copy(
            update={"utterance_id": benchmark.entries[0].utterance_id}
        )
    elif leak_kind == "speaker":
        train = train.model_copy(
            update={
                "speaker": train.speaker.model_copy(
                    update={"speaker_id": benchmark.entries[0].speaker_id}
                )
            }
        )
    elif leak_kind == "audio":
        train = train.model_copy(update={"audio": benchmark_record.audio})
    elif leak_kind == "surface":
        train = train.model_copy(
            update={
                "ground_truth": train.ground_truth.model_copy(
                    update={"surface_text": "국밥 하나 주이소!"}
                )
            }
        )
    assignments = TrainingSplitAssignments(
        train_utterance_ids=(train.utterance_id,),
        validation_utterance_ids=(validation.utterance_id,),
    )

    with pytest.raises(TrainingDatasetValidationError, match=message):
        build_training_dataset(
            dataset_id=f"{leak_kind}-leak-test",
            dataset_version="0.1.0",
            records=(train, validation),
            assignments=assignments,
            benchmark_manifests=(benchmark,),
        )


def test_required_frozen_benchmark_must_be_checked(tmp_path: Path) -> None:
    train, validation, assignments = valid_training_fixture(tmp_path)

    with pytest.raises(TrainingDatasetValidationError, match="required frozen benchmark"):
        build_training_dataset(
            dataset_id="missing-benchmark-test",
            dataset_version="0.1.0",
            records=(train, validation),
            assignments=assignments,
            benchmark_manifests=(),
        )


def test_repeated_surface_text_crossing_train_validation_is_rejected(
    tmp_path: Path,
) -> None:
    train, validation, assignments = valid_training_fixture(tmp_path)
    validation = validation.model_copy(
        update={
            "ground_truth": validation.ground_truth.model_copy(
                update={"surface_text": train.ground_truth.surface_text}
            )
        }
    )
    with pytest.raises(
        TrainingDatasetValidationError,
        match="exact Surface transcript crosses train/validation",
    ):
        build_training_dataset(
            dataset_id="repeated-surface-test",
            dataset_version="0.1.0",
            records=(train, validation),
            assignments=assignments,
            benchmark_manifests=(frozen_benchmark(tmp_path),),
        )


def test_repeated_surface_text_within_train_is_reported_as_warning(
    tmp_path: Path,
) -> None:
    train, validation, _assignments = valid_training_fixture(tmp_path)
    second_train = make_record(
        tmp_path,
        index=3,
        speaker_id="second-train-speaker",
        surface_text=train.ground_truth.surface_text,
        frequency_hz=440,
    )
    assignments = TrainingSplitAssignments(
        train_utterance_ids=(train.utterance_id, second_train.utterance_id),
        validation_utterance_ids=(validation.utterance_id,),
    )
    _manifest, report = build_training_dataset(
        dataset_id="repeated-train-surface-test",
        dataset_version="0.1.0",
        records=(train, second_train, validation),
        assignments=assignments,
        benchmark_manifests=(frozen_benchmark(tmp_path),),
    )

    assert report.passed is True
    assert len(report.duplicate_surface_text_groups) == 1
    assert report.duplicate_surface_text_groups[0].splits == (DatasetSplit.TRAIN,)


def test_training_label_review_is_append_only_and_frozen_records_are_blocked(
    tmp_path: Path,
) -> None:
    storage = LabStorage(tmp_path / "lab")
    record = make_record(
        tmp_path,
        index=1,
        speaker_id="review-speaker",
        surface_text="밥 묵었나",
        frequency_hz=210,
        label_status=ReviewStatus.CANDIDATE,
    )
    storage.save_utterance(record)

    reviewed = review_training_label(
        storage,
        utterance_id=record.utterance_id,
        reviewer_id="reviewer-001",
        status=ReviewStatus.APPROVED,
        reason="원음과 Surface transcript 대조 완료",
    )

    assert reviewed.ground_truth.label_status is ReviewStatus.APPROVED
    assert reviewed.ground_truth.label_version == "label_v1"
    revisions = storage.list_label_revisions(record.utterance_id)
    assert len(revisions) == 1
    assert revisions[0].previous.label_status is ReviewStatus.CANDIDATE
    assert revisions[0].updated.label_status is ReviewStatus.APPROVED

    storage.save_manifest(
        build_manifest(
            benchmark_id="busan-surface-v0",
            benchmark_version="1.0.0",
            records=(reviewed,),
            split=DatasetSplit.TEST,
        )
    )
    with pytest.raises(ValueError, match="frozen benchmark"):
        review_training_label(
            storage,
            utterance_id=record.utterance_id,
            reviewer_id="reviewer-002",
            status=ReviewStatus.HUMAN_REVIEWED,
            reason=None,
        )


def test_training_cli_and_schema_exports_are_reproducible(tmp_path: Path) -> None:
    data_root = tmp_path / "lab"
    storage = LabStorage(data_root)
    train, validation, assignments = valid_training_fixture(tmp_path)
    storage.save_utterance(train)
    storage.save_utterance(validation)
    storage.save_manifest(frozen_benchmark(tmp_path))
    assignments_path = tmp_path / "assignments.json"
    assignments_path.write_text(
        assignments.model_dump_json(indent=2),
        encoding="utf-8",
    )

    assert main(
        [
            "create-training-dataset",
            "--dataset-id",
            "busan-asr-training-pilot-v0",
            "--dataset-version",
            "0.1.0",
            "--assignments",
            str(assignments_path),
            "--data-root",
            str(data_root),
        ]
    ) == 0
    assert main(
        [
            "validate-training-dataset",
            "--dataset-id",
            "busan-asr-training-pilot-v0",
            "--dataset-version",
            "0.1.0",
            "--data-root",
            str(data_root),
        ]
    ) == 0
    output = tmp_path / "training-export.zip"
    assert main(
        [
            "export-training-dataset",
            "--dataset-id",
            "busan-asr-training-pilot-v0",
            "--dataset-version",
            "0.1.0",
            "--output",
            str(output),
            "--data-root",
            str(data_root),
        ]
    ) == 0
    assert output.is_file()

    schema_dir = tmp_path / "schemas"
    export_schemas(schema_dir)
    assert {
        "training-dataset.schema.json",
        "training-dataset-validation-report.schema.json",
        "training-export-record.schema.json",
        "training-split-assignments.schema.json",
    }.issubset(path.name for path in schema_dir.iterdir())
