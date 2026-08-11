#!/usr/bin/env python3
"""Run fixed-revision Nemotron pretrained inference on an Audio Lab export ZIP."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import math
import os
import platform
import random
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adapter import SpeechModelAdapter, TransformersNemotronAdapter
from contract import (
    ContractError,
    ValidatedBenchmark,
    extract_validated_audio,
    load_jsonl,
    validate_benchmark_package,
    validate_nemotron_prediction_metadata,
    validate_prediction_documents,
)

OUTPUT_NAMES = (
    "predictions.jsonl",
    "execution_metadata.json",
    "inference_summary.json",
    "run.log",
)
FULL_SHA_LENGTH = 40
MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
MODEL_REVISION = "f3d333391852ba876df169dcc9ba902d25b6ab0b"
EXPERIMENT_ID = "task-003b-nemotron-3.5-asr-streaming-0.6b-pretrained-v0"


class IncompleteRunError(RuntimeError):
    """Raised after preserving outputs for a run with failed utterances."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-package", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the frozen ZIP and configuration without loading a model",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        benchmark = validate_from_config(args.benchmark_package, config)
        if args.validate_only:
            print(json.dumps(benchmark_summary(benchmark), indent=2, ensure_ascii=False))
            return 0
        run_inference(
            benchmark_package=args.benchmark_package,
            benchmark=benchmark,
            config=config,
            output_dir=args.output_dir,
        )
    except (ContractError, IncompleteRunError, OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


def load_config(path: Path) -> dict[str, Any]:
    text = path.expanduser().read_text(encoding="utf-8")
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as error:
            raise ContractError("non-JSON YAML requires PyYAML from requirements.txt") from error
        document = yaml.safe_load(text)
    if not isinstance(document, dict):
        raise ContractError("configuration must contain one object")
    _validate_config(document)
    return document


def validate_from_config(package_path: Path, config: Mapping[str, Any]) -> ValidatedBenchmark:
    benchmark = _mapping(config, "benchmark")
    return validate_benchmark_package(
        package_path,
        expected_benchmark_id=_string(benchmark, "id"),
        expected_benchmark_version=_string(benchmark, "version"),
        expected_utterances=_positive_int(benchmark, "expected_utterances"),
        schema_path=(
            Path(__file__).resolve().parent / "schemas" / "benchmark_manifest.schema.json"
        ),
    )


def run_inference(
    *,
    benchmark_package: Path,
    benchmark: ValidatedBenchmark,
    config: Mapping[str, Any],
    output_dir: Path,
    adapter_factory: Callable[..., SpeechModelAdapter] = TransformersNemotronAdapter,
) -> None:
    output_dir = output_dir.expanduser().resolve()
    _prepare_output_directory(output_dir)
    logger = _create_logger(output_dir / "run.log")
    started_at = _utc_now()
    model_config = _mapping(config, "model")
    runtime_config = _mapping(config, "runtime")
    _set_random_seed(_positive_or_zero_int(runtime_config, "random_seed"))
    logger.info("validated benchmark %s@%s", benchmark.benchmark_id, benchmark.benchmark_version)
    logger.info("loading fixed model revision %s", _string(model_config, "revision"))

    cache_dir_raw = runtime_config.get("cache_dir")
    cache_dir = Path(cache_dir_raw) if isinstance(cache_dir_raw, str) else None
    adapter = adapter_factory(
        model_id=_string(model_config, "id"),
        revision=_string(model_config, "revision"),
        target_language=_string(model_config, "target_language"),
        cache_dir=cache_dir,
    )
    warmup_ms = adapter.warmup()
    logger.info("synthetic warm-up completed in %.2f ms", warmup_ms)

    predictions_path = output_dir / "predictions.jsonl"
    metrics: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    successful_ids: list[str] = []
    total_inference_ms = 0.0
    with tempfile.TemporaryDirectory(prefix="task-003b-audio-") as temporary:
        audio_by_id = extract_validated_audio(
            benchmark_package,
            benchmark,
            Path(temporary),
        )
        inference_started = time.perf_counter()
        with predictions_path.open("x", encoding="utf-8") as prediction_file:
            for entry in benchmark.entries:
                try:
                    adapter.synchronize()
                    utterance_started = time.perf_counter()
                    transcript = adapter.transcribe(audio_by_id[entry.utterance_id])
                    adapter.synchronize()
                    latency_ms = (time.perf_counter() - utterance_started) * 1000
                    real_time_factor = latency_ms / entry.duration_ms
                    prediction = _prediction_document(
                        entry=entry,
                        transcript=transcript,
                        latency_ms=latency_ms,
                        config=config,
                        adapter=adapter,
                    )
                    prediction_file.write(
                        json.dumps(prediction, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                    prediction_file.flush()
                    os.fsync(prediction_file.fileno())
                    successful_ids.append(entry.utterance_id)
                    metrics.append(
                        {
                            "utterance_id": entry.utterance_id,
                            "audio_duration_ms": round(entry.duration_ms, 3),
                            "latency_ms": round(latency_ms, 3),
                            "real_time_factor": round(real_time_factor, 6),
                        }
                    )
                    logger.info(
                        "transcribed %s in %.2f ms (RTF %.4f)",
                        entry.utterance_id,
                        latency_ms,
                        real_time_factor,
                    )
                except Exception as error:  # preserve per-utterance evidence before failing the run
                    failures.append(
                        {
                            "utterance_id": entry.utterance_id,
                            "error_type": type(error).__name__,
                            "message": _sanitize_error_message(str(error)),
                        }
                    )
                    logger.exception("failed utterance %s", entry.utterance_id)
        total_inference_ms = (time.perf_counter() - inference_started) * 1000
    completed_at = _utc_now()

    summary = _inference_summary(
        benchmark=benchmark,
        successful_ids=successful_ids,
        failures=failures,
        model_load_ms=adapter.model_load_ms,
        warmup_ms=warmup_ms,
        total_inference_ms=total_inference_ms,
        metrics=metrics,
    )
    metadata = _execution_metadata(
        started_at=started_at,
        completed_at=completed_at,
        benchmark=benchmark,
        config=config,
        adapter=adapter,
    )
    _write_new_json(output_dir / "execution_metadata.json", metadata)
    _write_new_json(output_dir / "inference_summary.json", summary)

    if failures or len(successful_ids) != len(benchmark.entries):
        raise IncompleteRunError(
            "run is incomplete; partial evidence and failure details were preserved"
        )

    schema_path = Path(__file__).resolve().parent / "schemas" / "predictions.schema.json"
    records = validate_prediction_documents(
        load_jsonl(predictions_path),
        schema_path=schema_path,
        benchmark=benchmark,
    )
    validate_nemotron_prediction_metadata(
        records,
        model_id=adapter.model_id,
        resolved_revision=adapter.resolved_revision,
        target_language=_string(model_config, "target_language"),
    )
    logger.info("completed %d/%d utterances", len(successful_ids), len(benchmark.entries))


def benchmark_summary(benchmark: ValidatedBenchmark) -> dict[str, Any]:
    return {
        "benchmark_id": benchmark.benchmark_id,
        "benchmark_version": benchmark.benchmark_version,
        "benchmark_package_sha256": benchmark.package_sha256,
        "utterances": len(benchmark.entries),
        "sample_rates": sorted({entry.sample_rate for entry in benchmark.entries}),
        "channels": sorted({entry.channels for entry in benchmark.entries}),
        "status": "valid",
    }


def _prediction_document(
    *,
    entry: Any,
    transcript: str,
    latency_ms: float,
    config: Mapping[str, Any],
    adapter: SpeechModelAdapter,
) -> dict[str, Any]:
    model = _mapping(config, "model")
    return {
        "experiment_id": _string(config, "experiment_id"),
        "benchmark_id": _string(_mapping(config, "benchmark"), "id"),
        "benchmark_version": _string(_mapping(config, "benchmark"), "version"),
        "utterance_id": entry.utterance_id,
        "audio_sha256": entry.audio_sha256,
        "device": f"cuda:{adapter.device_name}",
        "inference_timestamp": _utc_now(),
        "result": {
            "schema_version": "1.0.0",
            "surface_text": transcript,
            "confidence": None,
            "confidence_supported": False,
            "latency_ms": round(latency_ms, 3),
            "model": {
                "name": adapter.model_id,
                "version": adapter.resolved_revision,
                "model_provider": _string(model, "provider"),
                "model_family": _string(model, "family"),
                "decoder_type": _string(model, "decoder_type"),
                "target_language": _string(model, "target_language"),
                "fine_tuned": False,
                "checkpoint_identifier": (
                    f"hf://{adapter.model_id}@{adapter.resolved_revision}"
                    f"#sha256:{adapter.model_artifact_sha256}"
                ),
            },
            "segments": [],
        },
    }


def _execution_metadata(
    *,
    started_at: str,
    completed_at: str,
    benchmark: ValidatedBenchmark,
    config: Mapping[str, Any],
    adapter: SpeechModelAdapter,
) -> dict[str, Any]:
    torch_info = _torch_environment()
    model = _mapping(config, "model")
    runtime = _mapping(config, "runtime")
    return {
        "experiment_id": _string(config, "experiment_id"),
        "started_at": started_at,
        "completed_at": completed_at,
        "hostname": socket.gethostname(),
        "operating_system": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch_info["pytorch_version"],
        "cuda_runtime_version": torch_info["cuda_runtime_version"],
        "cuda_driver_version": torch_info["cuda_driver_version"],
        "cudnn_version": torch_info["cudnn_version"],
        "gpu_name": torch_info["gpu_names"][0] if torch_info["gpu_names"] else adapter.device_name,
        "gpu_count": torch_info["gpu_count"],
        "gpu_vram": [
            {"device_index": index, "bytes": value}
            for index, value in enumerate(torch_info["gpu_vram_bytes"])
        ],
        "gpu_vram_bytes": torch_info["gpu_vram_bytes"],
        "nemo_version": _package_version("nemo_toolkit"),
        "transformers_version": _package_version("transformers"),
        "model_id": adapter.model_id,
        "requested_revision": adapter.requested_revision,
        "resolved_revision": adapter.resolved_revision,
        "model_provider": _string(model, "provider"),
        "model_family": _string(model, "family"),
        "decoder_type": _string(model, "decoder_type"),
        "fine_tuned": False,
        "target_language": _string(model, "target_language"),
        "model_cache_path": adapter.model_cache_path,
        "model_artifact_hash": f"sha256:{adapter.model_artifact_sha256}",
        "generation_output_type": adapter.generation_output_type,
        "benchmark_id": benchmark.benchmark_id,
        "benchmark_version": benchmark.benchmark_version,
        "benchmark_package_sha256": benchmark.package_sha256,
        "inference_config": _mapping(config, "inference"),
        "random_seed": runtime.get("random_seed"),
        "git_commit": _git_commit(Path(__file__).resolve().parent),
    }


def _inference_summary(
    *,
    benchmark: ValidatedBenchmark,
    successful_ids: list[str],
    failures: list[dict[str, str]],
    model_load_ms: float,
    warmup_ms: float,
    total_inference_ms: float,
    metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    successful = set(successful_ids)
    expected_ids = {entry.utterance_id for entry in benchmark.entries}
    latencies = [float(item["latency_ms"]) for item in metrics]
    rtfs = [float(item["real_time_factor"]) for item in metrics]
    return {
        "status": "complete" if successful == expected_ids and not failures else "incomplete",
        "expected_utterances": len(benchmark.entries),
        "successful_utterances": len(successful_ids),
        "failed_utterances": len(failures),
        "missing_utterance_ids": sorted(expected_ids - successful),
        "duplicate_utterance_ids": sorted(_duplicates(successful_ids)),
        "model_load_ms": round(model_load_ms, 3),
        "warmup_ms": round(warmup_ms, 3),
        "total_inference_ms": round(total_inference_ms, 3),
        "latency_mean_ms": _mean(latencies),
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "rtf_mean": _mean(rtfs),
        "confidence_supported": False,
        "utterance_metrics": metrics,
        "failures": failures,
    }


def _validate_config(config: Mapping[str, Any]) -> None:
    _require_exact_keys(
        config,
        {"task_id", "experiment_id", "benchmark", "model", "inference", "runtime"},
        "configuration",
    )
    if config.get("task_id") != "TASK-003B":
        raise ContractError("task_id must be TASK-003B")
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError(f"experiment_id must be {EXPERIMENT_ID}")
    benchmark = _mapping(config, "benchmark")
    _require_exact_keys(benchmark, {"id", "version", "expected_utterances"}, "benchmark")
    _string(benchmark, "id")
    _string(benchmark, "version")
    _positive_int(benchmark, "expected_utterances")

    model = _mapping(config, "model")
    _require_exact_keys(
        model,
        {
            "id",
            "revision",
            "provider",
            "family",
            "decoder_type",
            "target_language",
            "fine_tuned",
        },
        "model",
    )
    revision = _string(model, "revision")
    if len(revision) != FULL_SHA_LENGTH or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ContractError("model.revision must be a full lowercase commit SHA")
    if model.get("id") != MODEL_ID:
        raise ContractError(f"model.id must be {MODEL_ID}")
    if revision != MODEL_REVISION:
        raise ContractError(f"model.revision must be the verified commit {MODEL_REVISION}")
    expected_model_values = {
        "provider": "NVIDIA",
        "family": "FastConformer-RNNT",
        "decoder_type": "RNNT",
        "target_language": "ko-KR",
        "fine_tuned": False,
    }
    for key, expected in expected_model_values.items():
        if model.get(key) != expected:
            raise ContractError(f"model.{key} must be {expected!r}")
    _string(model, "id")

    inference = _mapping(config, "inference")
    _require_exact_keys(
        inference,
        {
            "mode",
            "decoding",
            "max_symbols_per_step",
            "context_biasing",
            "word_boosting",
            "external_language_model",
            "reference_conditioning",
            "postprocessing_llm",
            "punctuation",
            "capitalization",
        },
        "inference",
    )
    if inference.get("mode") != "offline_per_utterance":
        raise ContractError("inference.mode must be offline_per_utterance")
    if inference.get("decoding") != "transformers_5.13_default_greedy_rnnt":
        raise ContractError("inference.decoding must be transformers_5.13_default_greedy_rnnt")
    if inference.get("max_symbols_per_step") != 10:
        raise ContractError("inference.max_symbols_per_step must match model config value 10")
    for key in (
        "context_biasing",
        "word_boosting",
        "external_language_model",
        "reference_conditioning",
        "postprocessing_llm",
    ):
        if inference.get(key) is not False:
            raise ContractError(f"inference.{key} must be false for the pretrained baseline")
    for key in ("punctuation", "capitalization"):
        if inference.get(key) != "model_default":
            raise ContractError(f"inference.{key} must be model_default")

    runtime = _mapping(config, "runtime")
    _require_exact_keys(runtime, {"cache_dir", "random_seed"}, "runtime")
    if runtime.get("cache_dir") is not None and not isinstance(runtime.get("cache_dir"), str):
        raise ContractError("runtime.cache_dir must be null or a path string")
    _positive_or_zero_int(runtime, "random_seed")


def _prepare_output_directory(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in OUTPUT_NAMES if (output_dir / name).exists()]
    if existing:
        raise ContractError(
            f"append-only output files already exist: {existing}; choose a new output directory"
        )


def _create_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"task003b-{path.parent.name}-{id(path)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s")
    formatter.converter = time.gmtime
    handler = logging.FileHandler(path, mode="x", encoding="utf-8")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def _write_new_json(path: Path, document: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _sanitize_error_message(message: str) -> str:
    home = str(Path.home())
    return message.replace(home, "<home>")


def _torch_environment() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {
            "pytorch_version": None,
            "cuda_runtime_version": None,
            "cuda_driver_version": None,
            "cudnn_version": None,
            "gpu_names": [],
            "gpu_count": 0,
            "gpu_vram_bytes": [],
        }
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return {
        "pytorch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cuda_driver_version": _nvidia_driver_version(),
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(gpu_count)],
        "gpu_count": gpu_count,
        "gpu_vram_bytes": [
            torch.cuda.get_device_properties(index).total_memory for index in range(gpu_count)
        ],
    }


def _nvidia_driver_version() -> str | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    versions = sorted({line.strip() for line in completed.stdout.splitlines() if line.strip()})
    return ",".join(versions) or None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit(directory: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 6) if values else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 6)
    interpolated = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return round(interpolated, 6)


def _duplicates(values: list[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def _mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ContractError(f"{key} must be an object")
    return value


def _string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{key} must be a non-empty string")
    return value


def _positive_int(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ContractError(f"{key} must be a positive integer")
    return value


def _positive_or_zero_int(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ContractError(f"{key} must be a non-negative integer")
    return value


def _require_exact_keys(
    mapping: Mapping[str, Any],
    expected: set[str],
    location: str,
) -> None:
    actual = set(mapping)
    if actual != expected:
        raise ContractError(
            f"{location} keys differ; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
