from __future__ import annotations

import importlib.util
import math
import wave
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "external_inference/nemotron_3_5/run_gate3_batch.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("gate3_batch_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_percentile_uses_linear_interpolation() -> None:
    runner = _load_runner()
    assert runner.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert math.isclose(runner.percentile([1.0, 2.0, 3.0], 0.95), 2.9)


def test_endpoint_probe_detects_appended_silence(tmp_path: Path) -> None:
    runner = _load_runner()
    path = tmp_path / "speech.wav"
    samples = array(
        "h",
        [int(math.sin(index / 10) * 8000) for index in range(16_000)],
    )
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(samples.tobytes())

    result = runner.endpoint_probe(path)

    assert result["correct"] is True
    assert result["early"] is False
    assert 780 <= result["delay_ms"] <= 820


def test_batch_runner_contains_realtime_and_lifecycle_probes() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "pace_realtime=True" in source
    assert "compare_offline=True" in source
    assert "cancel_after_chunks=2" in source
    assert "reset_tracker.reset()" in source
    assert "torch.cuda.memory_allocated()" in source
