from __future__ import annotations

import copy
import hashlib
import io
import json
import sys
import wave
import zipfile
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from adapter import TranscriptionTrace  # noqa: E402
from contract import (  # noqa: E402
    ContractError,
    load_jsonl,
    validate_benchmark_package,
    validate_nemotron_prediction_metadata,
    validate_prediction_documents,
)
from run_diagnostics import (  # noqa: E402
    DIAGNOSTIC_TARGETS,
    run_diagnostics,
)
from run_inference import (  # noqa: E402
    IncompleteRunError,
    load_config,
    run_inference,
    validate_from_config,
)
from validate_diagnostics import validate_diagnostics  # noqa: E402

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
REVISION = "f3d333391852ba876df169dcc9ba902d25b6ab0b"
PREDICTION_SCHEMA = PACKAGE_ROOT / "schemas" / "predictions.schema.json"


def _wav_bytes(
    *,
    sample_rate: int = 16_000,
    channels: int = 1,
    frames: int = 160,
    sample_value: int = 0,
) -> bytes:
    payload = io.BytesIO()
    with wave.open(payload, "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        sample = int(sample_value).to_bytes(2, byteorder="little", signed=True)
        audio.writeframes(sample * frames * channels)
    return payload.getvalue()


def _entry(
    index: int,
    audio: bytes,
    *,
    utterance_id: str | None = None,
    path: str | None = None,
    audio_hash: str | None = None,
) -> dict[str, Any]:
    digest = audio_hash or hashlib.sha256(audio).hexdigest()
    return {
        "utterance_id": utterance_id or str(uuid4()),
        "speaker_id": f"speaker-{index}",
        "split": "test",
        "original_audio_sha256": hashlib.sha256(f"original-{index}".encode()).hexdigest(),
        "derived_audio_sha256": digest,
        "derived_audio_path": path or f"derived/asr_16k_mono/{index}/sample.wav",
        "lineage_audio_sha256s": [],
        "surface_text": "reference is evaluation-only",
        "normalized_meaning": None,
        "dialect_expressions": [],
    }


def _write_package(
    path: Path,
    entries_with_audio: list[tuple[dict[str, Any], bytes | None]],
    *,
    benchmark_version: str = "1.0.0",
    extra_members: dict[str, bytes] | None = None,
) -> Path:
    manifest = {
        "schema_version": "1.0.0",
        "benchmark_id": "busan-surface-v0",
        "benchmark_version": benchmark_version,
        "created_at": "2026-07-30T00:00:00Z",
        "frozen": True,
        "entries": [entry for entry, _audio in entries_with_audio],
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("benchmark.json", json.dumps(manifest))
        for entry, audio in entries_with_audio:
            if audio is not None:
                archive.writestr(entry["derived_audio_path"], audio)
        for name, payload in (extra_members or {}).items():
            archive.writestr(name, payload)
    return path


def _validate(path: Path, expected: int = 1):
    return validate_benchmark_package(
        path,
        expected_benchmark_id="busan-surface-v0",
        expected_benchmark_version="1.0.0",
        expected_utterances=expected,
        schema_path=PACKAGE_ROOT / "schemas" / "benchmark_manifest.schema.json",
    )


def _prediction(
    entry: Any,
    *,
    family: str = "FastConformer-RNNT",
    decoder: str = "RNNT",
    target_language: str | None = "ko-KR",
    surface_text: str = "모델 원문",
) -> dict[str, Any]:
    model = {
        "name": MODEL_ID,
        "version": REVISION,
        "model_provider": "NVIDIA",
        "model_family": family,
        "decoder_type": decoder,
        "fine_tuned": False,
        "checkpoint_identifier": f"hf://{MODEL_ID}@{REVISION}",
    }
    if target_language is not None:
        model["target_language"] = target_language
    return {
        "experiment_id": "task-003b-nemotron-3.5-asr-streaming-0.6b-pretrained-v0",
        "benchmark_id": "busan-surface-v0",
        "benchmark_version": "1.0.0",
        "utterance_id": entry.utterance_id,
        "audio_sha256": entry.audio_sha256,
        "device": "cuda:fixture",
        "inference_timestamp": "2026-07-30T00:00:00Z",
        "result": {
            "schema_version": "1.0.0",
            "surface_text": surface_text,
            "confidence": None,
            "confidence_supported": False,
            "latency_ms": 1.0,
            "model": model,
            "segments": [],
        },
    }


def test_valid_benchmark_zip_and_audio_contract(tmp_path: Path) -> None:
    audio = _wav_bytes()
    package = _write_package(tmp_path / "benchmark.zip", [(_entry(1, audio), audio)])

    benchmark = _validate(package)

    assert len(benchmark.entries) == 1
    assert benchmark.entries[0].sample_rate == 16_000
    assert benchmark.entries[0].channels == 1
    assert benchmark.entries[0].duration_ms == pytest.approx(10.0)


def test_audio_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    audio = _wav_bytes()
    entry = _entry(1, audio, audio_hash="a" * 64)
    package = _write_package(tmp_path / "benchmark.zip", [(entry, audio)])

    with pytest.raises(ContractError, match="hash mismatch"):
        _validate(package)


def test_missing_audio_is_rejected(tmp_path: Path) -> None:
    audio = _wav_bytes()
    package = _write_package(tmp_path / "benchmark.zip", [(_entry(1, audio), None)])

    with pytest.raises(ContractError, match="missing"):
        _validate(package)


def test_duplicate_utterance_id_is_rejected(tmp_path: Path) -> None:
    audio_one = _wav_bytes(frames=160)
    audio_two = _wav_bytes(frames=320)
    utterance_id = str(uuid4())
    entries = [
        (_entry(1, audio_one, utterance_id=utterance_id), audio_one),
        (_entry(2, audio_two, utterance_id=utterance_id), audio_two),
    ]
    package = _write_package(tmp_path / "benchmark.zip", entries)

    with pytest.raises(ContractError, match="duplicate utterance"):
        _validate(package, expected=2)


def test_path_traversal_is_rejected(tmp_path: Path) -> None:
    audio = _wav_bytes()
    entry = _entry(1, audio, path="../escape.wav")
    package = _write_package(tmp_path / "benchmark.zip", [(entry, audio)])

    with pytest.raises(ContractError, match="unsafe relative path"):
        _validate(package)


def test_wrong_benchmark_version_is_rejected(tmp_path: Path) -> None:
    audio = _wav_bytes()
    package = _write_package(
        tmp_path / "benchmark.zip",
        [(_entry(1, audio), audio)],
        benchmark_version="9.9.9",
    )

    with pytest.raises(ContractError, match="unexpected benchmark version"):
        _validate(package)


@pytest.mark.parametrize(
    ("sample_rate", "channels", "message"),
    [(8_000, 1, "16000 Hz"), (16_000, 2, "mono")],
)
def test_wrong_audio_shape_is_rejected(
    tmp_path: Path,
    sample_rate: int,
    channels: int,
    message: str,
) -> None:
    audio = _wav_bytes(sample_rate=sample_rate, channels=channels)
    package = _write_package(tmp_path / "benchmark.zip", [(_entry(1, audio), audio)])

    with pytest.raises(ContractError, match=message):
        _validate(package)


def test_empty_audio_is_rejected(tmp_path: Path) -> None:
    audio = _wav_bytes(frames=0)
    package = _write_package(tmp_path / "benchmark.zip", [(_entry(1, audio), audio)])

    with pytest.raises(ContractError, match="no frames"):
        _validate(package)


def test_manifest_and_zip_file_count_must_match(tmp_path: Path) -> None:
    audio = _wav_bytes()
    package = _write_package(
        tmp_path / "benchmark.zip",
        [(_entry(1, audio), audio)],
        extra_members={"unmanifested.wav": audio},
    )

    with pytest.raises(ContractError, match="unmanifested"):
        _validate(package)


def test_rnnt_prediction_schema_metadata_and_empty_transcript(tmp_path: Path) -> None:
    audio = _wav_bytes()
    package = _write_package(tmp_path / "benchmark.zip", [(_entry(1, audio), audio)])
    benchmark = _validate(package)
    prediction = _prediction(benchmark.entries[0], surface_text="")

    records = validate_prediction_documents(
        [prediction],
        schema_path=PREDICTION_SCHEMA,
        benchmark=benchmark,
    )
    validate_nemotron_prediction_metadata(
        records,
        model_id=MODEL_ID,
        resolved_revision=REVISION,
        target_language="ko-KR",
    )

    assert records[0]["result"]["surface_text"] == ""
    assert records[0]["result"]["segments"] == []


def test_unknown_prediction_field_is_rejected_by_strict_schema(tmp_path: Path) -> None:
    audio = _wav_bytes()
    package = _write_package(tmp_path / "benchmark.zip", [(_entry(1, audio), audio)])
    benchmark = _validate(package)
    prediction = _prediction(benchmark.entries[0])
    prediction["result"]["decoder_state"] = {}

    with pytest.raises(ContractError, match="Additional properties"):
        validate_prediction_documents([prediction], schema_path=PREDICTION_SCHEMA)


def test_confidence_support_contract_is_enforced(tmp_path: Path) -> None:
    audio = _wav_bytes()
    package = _write_package(tmp_path / "benchmark.zip", [(_entry(1, audio), audio)])
    benchmark = _validate(package)
    prediction = _prediction(benchmark.entries[0])
    prediction["result"]["confidence"] = 0.9

    with pytest.raises(ContractError, match="confidence_supported"):
        validate_prediction_documents([prediction])


def test_existing_ctc_prediction_without_target_language_remains_compatible(
    tmp_path: Path,
) -> None:
    audio = _wav_bytes()
    package = _write_package(tmp_path / "benchmark.zip", [(_entry(1, audio), audio)])
    benchmark = _validate(package)
    prediction = _prediction(
        benchmark.entries[0],
        family="Conformer-CTC",
        decoder="CTC greedy",
        target_language=None,
    )

    records = validate_prediction_documents(
        [prediction],
        schema_path=PREDICTION_SCHEMA,
        benchmark=benchmark,
    )

    assert "target_language" not in records[0]["result"]["model"]


def test_expected_utterance_count_mismatch_is_rejected(tmp_path: Path) -> None:
    audio = _wav_bytes()
    package = _write_package(tmp_path / "benchmark.zip", [(_entry(1, audio), audio)])

    with pytest.raises(ContractError, match="expected 10"):
        validate_from_config(package, load_config(PACKAGE_ROOT / "config.example.yaml"))


class FakeAdapter:
    model_id = MODEL_ID
    requested_revision = REVISION
    resolved_revision = REVISION
    model_artifact_sha256 = "b" * 64
    model_cache_path = "<model-cache>/fixture"
    model_load_ms = 2.0
    device_name = "fixture-gpu"
    generation_output_type = "FixtureGenerateOutput"

    def __init__(self, *, fail_path: str | None = None, **_kwargs: Any) -> None:
        self.fail_path = fail_path

    def warmup(self) -> float:
        return 1.0

    def synchronize(self) -> None:
        return None

    def transcribe(self, audio_path: Path) -> str:
        if self.fail_path and self.fail_path in str(audio_path):
            raise RuntimeError("fixture failure")
        return ""


class FakeDiagnosticAdapter(FakeAdapter):
    def transcribe_with_trace(self, audio_path: Path) -> TranscriptionTrace:
        transcript = (
            "지금 뭐 하노"
            if DIAGNOSTIC_TARGETS[3]["utterance_id"] in str(audio_path)
            else ""
        )
        token_ids = [1, 2, 3] if transcript else [0]
        return TranscriptionTrace(
            processor_call={
                "sampling_rate": 16_000,
                "language": "ko-KR",
                "return_tensors": "pt",
                "is_streaming_passed": False,
                "is_first_audio_chunk_passed": False,
            },
            processor_inputs={
                "type": "BatchFeature",
                "keys": ["input_features"],
                "fields": {
                    "input_features": {
                        "type": "Tensor",
                        "shape": [1, 80, 10],
                        "dtype": "torch.float32",
                        "device": "cpu",
                    }
                },
            },
            model_generate_call={
                "api": "model.generate",
                "return_dict_in_generate": True,
                "streamer_passed": False,
                "streaming_input_generator_passed": False,
                "is_streaming_passed_to_processor": False,
                "input_keys": ["input_features"],
            },
            raw_model_output={
                "type": "FixtureGenerateOutput",
                "keys": ["sequences"],
                "fields": {
                    "sequences": {
                        "type": "Tensor",
                        "shape": [1, len(token_ids)],
                        "dtype": "torch.int64",
                        "device": "cuda:0",
                        "values": [token_ids],
                    }
                },
            },
            decoded_with_special_tokens=f"<s>{transcript}</s>",
            special_decode_error=None,
            decoded_transcript=transcript,
            batch_decoded_transcripts=[transcript],
            batch_decode_error=None,
            adapter_transcript=transcript,
            adapter_transformation="identity_no_postprocessing",
            extraction_error=None,
        )


def test_fake_model_success_writes_importable_empty_transcript(tmp_path: Path) -> None:
    audio = _wav_bytes()
    package = _write_package(tmp_path / "benchmark.zip", [(_entry(1, audio), audio)])
    benchmark = _validate(package)
    config = copy.deepcopy(load_config(PACKAGE_ROOT / "config.example.yaml"))
    config["benchmark"]["expected_utterances"] = 1
    output_dir = tmp_path / "output"

    run_inference(
        benchmark_package=package,
        benchmark=benchmark,
        config=config,
        output_dir=output_dir,
        adapter_factory=FakeAdapter,
    )

    predictions = load_jsonl(output_dir / "predictions.jsonl")
    assert predictions[0]["result"]["surface_text"] == ""
    summary = json.loads((output_dir / "inference_summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["successful_utterances"] == 1


def test_partial_failure_writes_incomplete_summary_and_fails_run(tmp_path: Path) -> None:
    audio_one = _wav_bytes(frames=160)
    audio_two = _wav_bytes(frames=320)
    entry_one = _entry(1, audio_one)
    entry_two = _entry(2, audio_two)
    package = _write_package(
        tmp_path / "benchmark.zip",
        [(entry_one, audio_one), (entry_two, audio_two)],
    )
    benchmark = _validate(package, expected=2)
    config = copy.deepcopy(load_config(PACKAGE_ROOT / "config.example.yaml"))
    config["benchmark"]["expected_utterances"] = 2
    output_dir = tmp_path / "output"

    def factory(**kwargs: Any) -> FakeAdapter:
        return FakeAdapter(fail_path=entry_two["derived_audio_path"], **kwargs)

    with pytest.raises(IncompleteRunError, match="incomplete"):
        run_inference(
            benchmark_package=package,
            benchmark=benchmark,
            config=config,
            output_dir=output_dir,
            adapter_factory=factory,
        )

    summary = json.loads((output_dir / "inference_summary.json").read_text())
    assert summary["status"] == "incomplete"
    assert summary["successful_utterances"] == 1
    assert summary["failed_utterances"] == 1
    assert summary["missing_utterance_ids"] == [entry_two["utterance_id"]]


def _diagnostic_fixture(tmp_path: Path) -> tuple[Path, Any, dict[str, Any]]:
    entries_with_audio: list[tuple[dict[str, Any], bytes]] = []
    for index, target in enumerate(DIAGNOSTIC_TARGETS, start=1):
        audio = _wav_bytes(
            frames=160 + index,
            sample_value=100 + index,
        )
        utterance_id = str(target["utterance_id"])
        entry = _entry(
            index,
            audio,
            utterance_id=utterance_id,
            path=f"derived/asr_16k_mono/{utterance_id}/sample.wav",
        )
        entries_with_audio.append((entry, audio))
    package = _write_package(tmp_path / "benchmark.zip", entries_with_audio)
    benchmark = _validate(package, expected=4)
    config = copy.deepcopy(load_config(PACKAGE_ROOT / "config.example.yaml"))
    config["benchmark"]["expected_utterances"] = 4
    return package, benchmark, config


def test_task_003c_fake_diagnostics_capture_raw_and_adapter_evidence(
    tmp_path: Path,
) -> None:
    package, benchmark, config = _diagnostic_fixture(tmp_path)
    output_dir = tmp_path / "task-003c-output"

    run_diagnostics(
        benchmark_package=package,
        benchmark=benchmark,
        config=config,
        output_dir=output_dir,
        adapter_factory=FakeDiagnosticAdapter,
    )

    summary = validate_diagnostics(output_dir)
    assert summary["status"] == "complete"
    assert summary["captured_target_count"] == 4
    assert (
        summary["diagnostic_assessment"]["classification"]
        == "MODEL_RETURNED_EMPTY"
    )
    assert len(summary["targets"]) == 4
    assert [target["adapter_output_empty"] for target in summary["targets"]] == [
        True,
        True,
        True,
        False,
    ]
    probes = json.loads((output_dir / "audio_probe.json").read_text())["targets"]
    assert all(probe["sample_rate_hz"] == 16_000 for probe in probes)
    assert all(probe["channels"] == 1 for probe in probes)
    assert all(probe["hash_matches_manifest"] for probe in probes)
    assert all(probe["waveform_nonempty"] for probe in probes)
    raw = load_jsonl(output_dir / "raw_model_output.jsonl")
    sequence = raw[0]["raw_model_output"]["fields"]["sequences"]
    assert sequence["values"] == [[0]]
    traces = load_jsonl(output_dir / "adapter_trace.jsonl")
    assert traces[0]["decoded_with_special_tokens"] == "<s></s>"
    assert traces[-1]["decoded_transcript"] == "지금 뭐 하노"
    assert not (output_dir / "predictions.jsonl").exists()


def test_task_003c_requires_all_four_fixed_utterance_ids(tmp_path: Path) -> None:
    package, benchmark, config = _diagnostic_fixture(tmp_path)
    incomplete_benchmark = type(benchmark)(
        benchmark_id=benchmark.benchmark_id,
        benchmark_version=benchmark.benchmark_version,
        package_sha256=benchmark.package_sha256,
        entries=benchmark.entries[:-1],
    )

    with pytest.raises(ContractError, match="target utterances are missing"):
        run_diagnostics(
            benchmark_package=package,
            benchmark=incomplete_benchmark,
            config=config,
            output_dir=tmp_path / "task-003c-output",
            adapter_factory=FakeDiagnosticAdapter,
        )


def test_task_003c_validator_rejects_duplicate_trace_ids(tmp_path: Path) -> None:
    package, benchmark, config = _diagnostic_fixture(tmp_path)
    output_dir = tmp_path / "task-003c-output"
    run_diagnostics(
        benchmark_package=package,
        benchmark=benchmark,
        config=config,
        output_dir=output_dir,
        adapter_factory=FakeDiagnosticAdapter,
    )
    trace_path = output_dir / "adapter_trace.jsonl"
    traces = load_jsonl(trace_path)
    traces[1]["utterance_id"] = traces[0]["utterance_id"]
    trace_path.write_text(
        "\n".join(json.dumps(trace, ensure_ascii=False) for trace in traces) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match="duplicate utterance IDs"):
        validate_diagnostics(output_dir)
