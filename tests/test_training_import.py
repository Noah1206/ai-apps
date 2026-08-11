from __future__ import annotations

from pathlib import Path

import pytest

from busan_lab.audio import AudioProcessor
from busan_lab.config import LabSettings
from busan_lab.manifest import build_manifest
from busan_lab.schemas.common import (
    ConsentRecord,
    DatasetSplit,
    ReviewStatus,
)
from busan_lab.schemas.training_import import TrainingRecordingReviewDecision
from busan_lab.schemas.utterance import (
    LinguisticGroundTruth,
    SpeakerContext,
    UtteranceRecord,
)
from busan_lab.storage import LabStorage
from busan_lab.training_import import (
    build_training_recording_import_plan,
    build_training_recording_review_queue,
    execute_training_recording_import,
    list_training_recording_import_summaries,
    review_training_recording,
)
from tests.helpers import make_wav_bytes


def _prompt_sheet(root: Path) -> Path:
    path = root / "SOLO_SPEAKER_300.md"
    path.write_text(
        "\n".join(
            (
                "- T004-S001: 마, 지금 출발하나?",
                "- T004-S002: 니 오늘 학교 가노?",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _recordings(root: Path) -> Path:
    directory = root / "recordings"
    directory.mkdir()
    (directory / "New Recording.m4a").write_bytes(
        make_wav_bytes(frequency_hz=220)
    )
    (directory / "New Recording 2.m4a").write_bytes(
        make_wav_bytes(frequency_hz=330)
    )
    return directory


def _frozen_benchmark(root: Path, storage: LabStorage):
    source = root / "benchmark.wav"
    source.write_bytes(make_wav_bytes(frequency_hz=440))
    audio = AudioProcessor(
        LabSettings.from_environment(storage.root),
        storage,
    ).process(source, source.name)
    record = UtteranceRecord(
        speaker=SpeakerContext(
            speaker_id="benchmark-speaker",
            region="Busan",
            device="fixture",
            environment="quiet",
        ),
        consent=ConsentRecord(
            storage_allowed=True,
            research_use_allowed=True,
            model_training_allowed=False,
        ),
        audio=audio,
        ground_truth=LinguisticGroundTruth(
            surface_text="국밥 하나 주이소",
            label_status=ReviewStatus.APPROVED,
        ),
    )
    manifest = build_manifest(
        benchmark_id="busan-surface-v0",
        benchmark_version="1.0.0",
        records=(record,),
        split=DatasetSplit.TEST,
    )
    storage.save_manifest(manifest)
    return manifest


def _consent() -> ConsentRecord:
    return ConsentRecord(
        storage_allowed=True,
        research_use_allowed=True,
        model_training_allowed=True,
    )


def test_training_recordings_plan_and_import_are_reproducible(tmp_path: Path) -> None:
    storage = LabStorage(tmp_path / "lab")
    settings = LabSettings.from_environment(storage.root)
    benchmark = _frozen_benchmark(tmp_path, storage)
    recordings = _recordings(tmp_path)
    plan = build_training_recording_import_plan(
        import_id="task-004-solo-speaker-001-v0",
        input_directory=recordings,
        prompt_sheet=_prompt_sheet(tmp_path),
        prompt_start=1,
        prompt_end=2,
        speaker_id="busan-train-speaker-001",
        region="Busan",
        device="Apple Voice Memos",
        recording_environment="quiet_room",
        consent=_consent(),
        benchmark_manifests=(benchmark,),
    )

    assert plan.passed is True
    assert [item.prompt_id for item in plan.items] == ["T004-S001", "T004-S002"]
    assert [item.source_filename for item in plan.items] == [
        "New Recording.m4a",
        "New Recording 2.m4a",
    ]

    manifest = execute_training_recording_import(
        plan=plan,
        input_directory=recordings,
        storage=storage,
        processor=AudioProcessor(settings, storage),
    )

    assert len(manifest.entries) == 2
    assert storage.load_training_recording_import(manifest.import_id) == manifest
    imported_records = tuple(
        storage.load_utterance(entry.utterance_id) for entry in manifest.entries
    )
    assert all(record.source == "import" for record in imported_records)
    assert all(record.dataset_split is DatasetSplit.TRAIN for record in imported_records)
    assert all(
        record.ground_truth.label_status is ReviewStatus.CANDIDATE
        for record in imported_records
    )
    assert all(record.consent.model_training_allowed for record in imported_records)
    assert all(record.audio.original.relative_path.endswith(".m4a") for record in imported_records)
    assert all(record.audio.derived.sample_rate_hz == 16_000 for record in imported_records)
    assert all(record.audio.derived.channels == 1 for record in imported_records)
    with pytest.raises(FileExistsError, match="already exists"):
        execute_training_recording_import(
            plan=plan,
            input_directory=recordings,
            storage=storage,
            processor=AudioProcessor(settings, storage),
        )

    summaries = list_training_recording_import_summaries(storage)
    assert [(item.import_id, item.entry_count) for item in summaries] == [
        (manifest.import_id, 2)
    ]
    initial_queue = build_training_recording_review_queue(
        storage,
        import_id=manifest.import_id,
    )
    assert initial_queue.candidate_count == 2
    assert [item.prompt_id for item in initial_queue.items] == [
        "T004-S001",
        "T004-S002",
    ]

    review_training_recording(
        storage,
        import_id=manifest.import_id,
        prompt_id="T004-S001",
        reviewer_id="labeler-001",
        decision=TrainingRecordingReviewDecision.APPROVE,
        notes="원음과 대조 완료",
    )
    review_training_recording(
        storage,
        import_id=manifest.import_id,
        prompt_id="T004-S002",
        reviewer_id="labeler-001",
        decision=TrainingRecordingReviewDecision.RERECORD,
        notes="앞 음절이 잘림",
    )
    reviewed_queue = build_training_recording_review_queue(
        storage,
        import_id=manifest.import_id,
    )
    assert reviewed_queue.reviewed_count == 2
    assert reviewed_queue.approved_count == 1
    assert reviewed_queue.rerecord_count == 1
    assert reviewed_queue.candidate_count == 0


def test_training_recording_plan_reports_missing_file_without_writing(
    tmp_path: Path,
) -> None:
    storage = LabStorage(tmp_path / "lab")
    benchmark = _frozen_benchmark(tmp_path, storage)
    recordings = _recordings(tmp_path)
    (recordings / "New Recording 2.m4a").unlink()

    plan = build_training_recording_import_plan(
        import_id="missing-file-v0",
        input_directory=recordings,
        prompt_sheet=_prompt_sheet(tmp_path),
        prompt_start=1,
        prompt_end=2,
        speaker_id="busan-train-speaker-001",
        region="Busan",
        device="Apple Voice Memos",
        recording_environment="quiet_room",
        consent=_consent(),
        benchmark_manifests=(benchmark,),
    )

    assert plan.passed is False
    assert any("missing recording numbers: [2]" in error for error in plan.errors)
    assert not storage.training_recording_import_exists(plan.import_id)


def test_training_recording_plan_can_report_an_entirely_empty_batch(
    tmp_path: Path,
) -> None:
    storage = LabStorage(tmp_path / "lab")
    benchmark = _frozen_benchmark(tmp_path, storage)
    recordings = tmp_path / "empty-recordings"
    recordings.mkdir()

    plan = build_training_recording_import_plan(
        import_id="empty-batch-v0",
        input_directory=recordings,
        prompt_sheet=_prompt_sheet(tmp_path),
        prompt_start=1,
        prompt_end=2,
        speaker_id="busan-train-speaker-001",
        region="Busan",
        device="Apple Voice Memos",
        recording_environment="quiet_room",
        consent=_consent(),
        benchmark_manifests=(benchmark,),
    )

    assert plan.passed is False
    assert plan.items == ()
    assert any("missing recording numbers: [1, 2]" in error for error in plan.errors)


def test_training_recording_plan_requires_all_explicit_consents(tmp_path: Path) -> None:
    storage = LabStorage(tmp_path / "lab")
    benchmark = _frozen_benchmark(tmp_path, storage)
    plan = build_training_recording_import_plan(
        import_id="missing-consent-v0",
        input_directory=_recordings(tmp_path),
        prompt_sheet=_prompt_sheet(tmp_path),
        prompt_start=1,
        prompt_end=2,
        speaker_id="busan-train-speaker-001",
        region="Busan",
        device="Apple Voice Memos",
        recording_environment="quiet_room",
        consent=ConsentRecord(
            storage_allowed=True,
            research_use_allowed=False,
            model_training_allowed=False,
        ),
        benchmark_manifests=(benchmark,),
    )

    assert plan.passed is False
    assert "explicit research-use consent is required" in plan.errors
    assert "explicit model-training consent is required" in plan.errors
