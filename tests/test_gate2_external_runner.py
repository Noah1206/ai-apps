from __future__ import annotations

import hashlib
import importlib.util
import json
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "external_inference/nemotron_3_5/run_gate2_nemo_suite.py"


def test_gate2_gpu_wrapper_reads_only_frozen_held_out_audio(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("gate2_gpu_wrapper", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    audio = tmp_path / "sample.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\0\0" * 160)
    audio_sha = hashlib.sha256(audio.read_bytes()).hexdigest()
    manifest = {
        "dataset_id": "busan-held-out-v0",
        "dataset_version": "1.0.0",
        "dataset_kind": "independent_busan_test",
        "frozen": True,
        "training_allowed": False,
        "checkpoint_selection_allowed": False,
        "entries": [
            {
                "utterance_id": "held-out-1",
                "audio_filepath": audio.name,
                "audio_sha256": audio_sha,
                "surface_text": "reference must not enter inference entries",
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded, entries = module.read_dataset(manifest_path)

    assert loaded == manifest
    assert entries == [
        {
            "utterance_id": "held-out-1",
            "audio_sha256": audio_sha,
            "audio_path": audio,
        }
    ]
    assert "surface_text" not in entries[0]
