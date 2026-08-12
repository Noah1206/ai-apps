from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build_gate3_engineering_benchmark.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("gate3_benchmark_builder", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selection_is_deterministic_balanced_and_duration_bounded() -> None:
    builder = _load_builder()
    rows = [
        {
            "utterance_id": f"u-{speaker}-{sample}",
            "speaker_id": f"speaker-{speaker}",
            "duration": duration,
        }
        for speaker in range(7)
        for sample, duration in enumerate((2.0, 3.0, 4.0, 8.0))
    ]
    arguments = {
        "seed": "fixed",
        "speaker_count": 5,
        "samples_per_speaker": 2,
        "min_duration_seconds": 2.5,
        "max_duration_seconds": 7.0,
    }

    first = builder.select_rows(rows, **arguments)
    second = builder.select_rows(list(reversed(rows)), **arguments)

    assert [row["utterance_id"] for row in first] == [row["utterance_id"] for row in second]
    assert len(first) == 10
    assert len({row["speaker_id"] for row in first}) == 5
    assert all(2.5 <= row["duration"] <= 7.0 for row in first)


def test_exclusion_keys_load_prior_utterances_and_audio_hashes(tmp_path: Path) -> None:
    builder = _load_builder()
    manifest = tmp_path / "prior.jsonl"
    manifest.write_text(
        json.dumps({"utterance_id": "u-1", "audio_sha256": "a" * 64}) + "\n",
        encoding="utf-8",
    )

    utterance_ids, audio_sha256s = builder.exclusion_keys([manifest])

    assert utterance_ids == {"u-1"}
    assert audio_sha256s == {"a" * 64}
