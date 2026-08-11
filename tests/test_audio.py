import wave
from pathlib import Path

import pytest

from busan_lab.audio import AudioProcessor, AudioValidationError, hash_file
from busan_lab.config import LabSettings
from busan_lab.schemas.audio import AudioRole
from busan_lab.storage import LabStorage
from tests.helpers import make_wav_bytes


def test_original_is_preserved_and_master_derivative_contracts_are_fixed(
    tmp_path: Path,
) -> None:
    storage = LabStorage(tmp_path / "lab")
    processor = AudioProcessor(LabSettings.from_environment(storage.root), storage)
    source_bytes = make_wav_bytes(sample_rate=44_100, channels=2)
    staged = storage.staging_dir / "source.wav"
    staged.write_bytes(source_bytes)

    bundle = processor.process(staged, "부산 원본.wav")

    original_path = storage.resolve(bundle.original.relative_path)
    assert bundle.master is not None
    master_path = storage.resolve(bundle.master.relative_path)
    asr_path = storage.resolve(bundle.derived.relative_path)
    assert original_path.read_bytes() == source_bytes
    assert bundle.original.sha256 == hash_file(original_path)
    assert bundle.master.sha256 == hash_file(master_path)
    assert bundle.master.parent_sha256 == bundle.original.sha256
    assert bundle.derived.sha256 == hash_file(asr_path)
    assert bundle.derived.parent_sha256 == bundle.master.sha256
    with wave.open(str(master_path), "rb") as wav:
        assert wav.getframerate() == 48_000
        assert wav.getnchannels() == 2
        assert wav.getsampwidth() == 3
    with wave.open(str(asr_path), "rb") as wav:
        assert wav.getframerate() == 16_000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
    expected_contracts = {
        AudioRole.ASR_16K_MONO: (16_000, 1, 2),
        AudioRole.PRONUNCIATION_24K_MONO: (24_000, 1, 2),
        AudioRole.TTS_48K_MONO: (48_000, 1, 3),
    }
    assert {asset.role for asset in bundle.derivatives} == set(expected_contracts)
    for role, (sample_rate, channels, sample_width) in expected_contracts.items():
        asset = bundle.asset_for_role(role)
        assert asset.parent_sha256 == bundle.master.sha256
        with wave.open(str(storage.resolve(asset.relative_path)), "rb") as wav:
            assert wav.getframerate() == sample_rate
            assert wav.getnchannels() == channels
            assert wav.getsampwidth() == sample_width


@pytest.mark.parametrize(
    ("source_channels", "expected_master_role"),
    [
        (1, AudioRole.MASTER_48K_MONO),
        (2, AudioRole.MASTER_48K_STEREO),
    ],
)
def test_48khz_wav_master_preserves_mono_or_stereo_contract(
    tmp_path: Path,
    source_channels: int,
    expected_master_role: AudioRole,
) -> None:
    storage = LabStorage(tmp_path / f"lab-{source_channels}")
    processor = AudioProcessor(LabSettings.from_environment(storage.root), storage)
    staged = storage.staging_dir / "master-source.wav"
    staged.write_bytes(make_wav_bytes(sample_rate=48_000, channels=source_channels))

    bundle = processor.process(staged, "master-source.wav")

    assert bundle.original.container == "wav"
    assert bundle.master is not None
    assert bundle.master.role is expected_master_role
    assert bundle.master.sample_rate_hz == 48_000
    assert bundle.master.channels == source_channels


def test_processing_same_bytes_is_deterministic(tmp_path: Path) -> None:
    storage = LabStorage(tmp_path / "lab")
    processor = AudioProcessor(LabSettings.from_environment(storage.root), storage)
    first = storage.staging_dir / "first.wav"
    second = storage.staging_dir / "second.wav"
    payload = make_wav_bytes()
    first.write_bytes(payload)
    second.write_bytes(payload)

    first_bundle = processor.process(first, first.name)
    second_bundle = processor.process(second, second.name)

    assert first_bundle.original.sha256 == second_bundle.original.sha256
    assert first_bundle.master is not None
    assert second_bundle.master is not None
    assert first_bundle.master.sha256 == second_bundle.master.sha256
    assert first_bundle.derived.sha256 == second_bundle.derived.sha256
    assert first_bundle.original.relative_path == second_bundle.original.relative_path
    assert [asset.sha256 for asset in first_bundle.derivatives] == [
        asset.sha256 for asset in second_bundle.derivatives
    ]


def test_non_audio_is_rejected_before_raw_preservation(tmp_path: Path) -> None:
    storage = LabStorage(tmp_path / "lab")
    processor = AudioProcessor(LabSettings.from_environment(storage.root), storage)
    staged = storage.staging_dir / "not-audio.bin"
    staged.write_bytes(b"not audio")

    with pytest.raises(AudioValidationError, match="not decodable audio"):
        processor.process(staged, "not-audio.bin")

    assert list(storage.raw_dir.rglob("*.*")) == []
