from __future__ import annotations

import hashlib
import importlib.util
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "external_inference/nemotron_3_5/run_gate3_streaming.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("gate3_streaming_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_wav(path: Path, *, sample_rate: int = 16_000, channels: int = 1) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\0\0" * sample_rate * channels)


def test_gate3_runner_enforces_pcm16_mono_16khz(tmp_path: Path) -> None:
    runner = _load_runner()
    valid = tmp_path / "valid.wav"
    _write_wav(valid)
    probe = runner.inspect_audio(valid)
    assert probe["duration_ms"] == 1000
    assert probe["sha256"] == hashlib.sha256(valid.read_bytes()).hexdigest()

    invalid = tmp_path / "invalid.wav"
    _write_wav(invalid, channels=2)
    with pytest.raises(ValueError, match="16 kHz mono"):
        runner.inspect_audio(invalid)


def test_gate3_runner_uses_only_audio_hash_to_join_surface_prediction(tmp_path: Path) -> None:
    runner = _load_runner()
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "audio_sha256": "a" * 64,
                "result": {"surface_text": "밥 묵었나", "reference_surface_text": "unused"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    assert runner.find_surface_prediction(predictions, "a" * 64) == "밥 묵었나"
    assert runner.find_surface_prediction(predictions, "b" * 64) is None


def test_gate3_runner_is_cache_aware_and_stepwise() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "CacheAwareStreamingAudioBuffer" in source
    assert "asr_model.conformer_stream_step(" in source
    assert '"compute_dtype": "float32"' in source
    assert "asr_model.set_inference_prompt(TARGET_LANGUAGE)" in source
    assert "get_initial_cache_state(batch_size=1)" in source
    assert "if events and not cancelled:" in source
    assert "final_event = tracker.observe(" in source
    assert '"runtime_candidate": "runtime-v5"' in source
    assert "streaming_buffer.append_audio(padded_audio" in source


def test_gate3_runner_appends_exact_zero_pcm_flush(tmp_path: Path) -> None:
    runner = _load_runner()
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x40" * 1600)

    padded = runner._read_pcm16_with_end_padding(source, 320.0)

    assert len(padded) == 1600 + 5120
    assert padded[0] == 0.5
    assert (padded[-5120:] == 0.0).all()


def test_gate3_runtime_config_freezes_streaming_only_flush(tmp_path: Path) -> None:
    runner = _load_runner()
    config = runner._runtime_config(tmp_path / "model.nemo")
    assert config["runtime_candidate"] == "runtime-v5"
    assert config["end_of_stream_padding_ms"] == 320.0
    assert config["end_of_stream_padding_policy"] == ("zero_pcm_streaming_only_offline_unpadded")
    assert config["end_of_stream_pacing_policy"] == (
        "process_synthetic_flush_immediately_after_explicit_eof"
    )


def test_gate3_realtime_pacing_does_not_wait_for_synthetic_flush() -> None:
    runner = _load_runner()

    assert runner._realtime_pace_target_ms(audio_end_ms=960.0, source_duration_ms=1000.0) == 960.0
    assert runner._realtime_pace_target_ms(audio_end_ms=1320.0, source_duration_ms=1000.0) == 1000.0


class _FakeFeatureBuffer:
    def __init__(self, frames: int) -> None:
        self.frames = frames

    def size(self, dimension: int) -> int:
        assert dimension == -1
        return self.frames


def test_gate3_runner_detects_unyieldable_residual_feature_tail() -> None:
    runner = _load_runner()
    streaming_buffer = SimpleNamespace(
        buffer_idx=160,
        buffer=_FakeFeatureBuffer(165),
        streaming_cfg=SimpleNamespace(chunk_size=[25, 32]),
        sampling_frames=[8, 8],
    )
    assert runner._streaming_buffer_has_next_chunk(streaming_buffer) is False

    streaming_buffer.buffer_idx = 150
    assert runner._streaming_buffer_has_next_chunk(streaming_buffer) is True
