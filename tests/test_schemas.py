from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from busan_lab.audio import AudioProcessor
from busan_lab.config import LabSettings
from busan_lab.manifest import build_manifest
from busan_lab.schemas.benchmark import BenchmarkEntry, BenchmarkManifest
from busan_lab.schemas.common import ConsentRecord, DatasetSplit
from busan_lab.schemas.utterance import (
    LinguisticGroundTruth,
    SpeakerContext,
    UtteranceRecord,
)
from busan_lab.storage import LabStorage
from tests.helpers import make_wav_bytes


def make_record(tmp_path: Path, speaker_id: str = "speaker-1") -> UtteranceRecord:
    storage = LabStorage(tmp_path / "lab")
    settings = LabSettings.from_environment(storage.root)
    source = tmp_path / f"{speaker_id}.wav"
    source.write_bytes(make_wav_bytes())
    audio = AudioProcessor(settings, storage).process(source, source.name)
    return UtteranceRecord(
        speaker=SpeakerContext(speaker_id=speaker_id, region="Busan"),
        consent=ConsentRecord(
            storage_allowed=True,
            research_use_allowed=True,
        ),
        audio=audio,
        ground_truth=LinguisticGroundTruth(
            surface_text="국밥 하나 주이소",
            normalized_meaning="국밥 하나 주세요",
        ),
    )


def test_surface_and_normalized_meaning_remain_distinct(tmp_path: Path) -> None:
    record = make_record(tmp_path)

    assert record.ground_truth.surface_text == "국밥 하나 주이소"
    assert record.ground_truth.normalized_meaning == "국밥 하나 주세요"
    dumped = record.model_dump()
    assert "surface_text" in dumped["ground_truth"]
    assert "normalized_meaning" in dumped["ground_truth"]


def test_unknown_schema_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        LinguisticGroundTruth.model_validate(
            {"surface_text": "주이소", "normalized_text": "주세요"}
        )


def test_manifest_rejects_speaker_crossing_splits(tmp_path: Path) -> None:
    record = make_record(tmp_path)
    base_entry = BenchmarkEntry(
        utterance_id=record.utterance_id,
        speaker_id=record.speaker.speaker_id,
        split=DatasetSplit.TRAIN,
        original_audio_sha256=record.audio.original.sha256,
        derived_audio_sha256=record.audio.derived.sha256,
        derived_audio_path=record.audio.derived.relative_path,
        surface_text=record.ground_truth.surface_text,
    )
    crossing_entry = base_entry.model_copy(
        update={
            "utterance_id": uuid4(),
            "split": DatasetSplit.TEST,
            "original_audio_sha256": "a" * 64,
            "derived_audio_sha256": "b" * 64,
        }
    )

    with pytest.raises(ValidationError, match="speaker"):
        BenchmarkManifest(
            benchmark_id="leak-test",
            benchmark_version="0.1",
            entries=(base_entry, crossing_entry),
        )


def test_build_manifest_requires_explicit_split(tmp_path: Path) -> None:
    record = make_record(tmp_path)

    with pytest.raises(ValueError, match="explicit"):
        build_manifest(
            benchmark_id="v0",
            benchmark_version="0.1",
            records=[record],
            split=DatasetSplit.UNASSIGNED,
        )
