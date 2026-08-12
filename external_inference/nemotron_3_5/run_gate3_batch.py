"""Evaluate the frozen Gate 3 engineering batch on one loaded NeMo model."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import wave
from array import array
from pathlib import Path
from statistics import fmean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from busan_lab.evaluation.metrics import character_error_rate  # noqa: E402
from busan_lab.schemas.streaming import Gate3Criteria, Gate3Evidence  # noqa: E402
from busan_lab.streaming import (  # noqa: E402
    EnergyEndpointDetector,
    StreamingTranscriptSession,
    assess_gate3,
)
from external_inference.nemotron_3_5.run_gate3_streaming import (  # noqa: E402
    DEFAULT_END_OF_STREAM_PADDING_MS,
    DEFAULT_MODEL_SHA256,
    _runtime_config,
    _stream_audio,
    _write_json,
    build_model_descriptor,
    inspect_audio,
    load_asr_model,
    sha256_file,
)

APPENDED_SILENCE_MS = 1200.0
ENDPOINT_EARLY_TOLERANCE_MS = 200.0
ENDPOINT_LATE_TOLERANCE_MS = 1000.0
CONFIDENCE_POLICY = "explicitly_unsupported_no_fabricated_values"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--criteria", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--stabilization-window", type=int, default=3)
    parser.add_argument(
        "--end-of-stream-padding-ms",
        type=float,
        default=DEFAULT_END_OF_STREAM_PADDING_MS,
    )
    return parser


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile from no values")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {
                "case_id",
                "speaker_id",
                "audio_filepath",
                "audio_sha256",
                "duration_seconds",
            }
            missing = required - row.keys()
            if missing:
                raise ValueError(f"manifest line {line_number} is missing {sorted(missing)}")
            rows.append(row)
    case_ids = [str(row["case_id"]) for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Gate 3 manifest contains duplicate case IDs")
    return rows


def endpoint_probe(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as audio:
        if (
            audio.getnchannels(),
            audio.getsampwidth(),
            audio.getframerate(),
            audio.getcomptype(),
        ) != (1, 2, 16_000, "NONE"):
            raise ValueError("endpoint probe requires 16 kHz mono PCM16 WAV")
        frame_count = audio.getnframes()
        samples = array("h", audio.readframes(frame_count))
    if sys.byteorder == "big":
        samples.byteswap()
    source_duration_ms = frame_count / 16_000 * 1000
    appended_samples = int(APPENDED_SILENCE_MS * 16_000 / 1000)
    samples.extend([0] * appended_samples)
    detector = EnergyEndpointDetector(
        speech_threshold_dbfs=-42,
        minimum_speech_ms=200,
        trailing_silence_ms=800,
    )
    frame_size = 320
    for start in range(0, len(samples), frame_size):
        frame = samples[start : start + frame_size]
        if not frame:
            continue
        mean_square = sum(float(value) ** 2 for value in frame) / len(frame)
        normalized_rms = min(1.0, math.sqrt(mean_square) / 32768)
        frame_end_ms = min(start + len(frame), len(samples)) / 16_000 * 1000
        if detector.observe(
            normalized_rms=normalized_rms,
            frame_duration_ms=len(frame) / 16_000 * 1000,
            frame_end_ms=frame_end_ms,
        ):
            break
    detected_ms = detector.endpoint_ms
    correct = bool(
        detected_ms is not None
        and source_duration_ms - ENDPOINT_EARLY_TOLERANCE_MS
        <= detected_ms
        <= source_duration_ms + ENDPOINT_LATE_TOLERANCE_MS
    )
    early = detected_ms is not None and detected_ms < (
        source_duration_ms - ENDPOINT_EARLY_TOLERANCE_MS
    )
    return {
        "source_duration_ms": source_duration_ms,
        "appended_silence_ms": APPENDED_SILENCE_MS,
        "detected_ms": detected_ms,
        "delay_ms": max(0.0, detected_ms - source_duration_ms) if detected_ms else None,
        "correct": correct,
        "early": early,
    }


def _aggregate_cer(pairs: list[tuple[str, str]]) -> float:
    total_edits = 0
    reference_characters = 0
    for reference, hypothesis in pairs:
        _cer, edits = character_error_rate(reference, hypothesis)
        total_edits += edits.substitutions + edits.deletions + edits.insertions
        reference_characters += edits.reference_characters
    if reference_characters == 0:
        return float(total_edits)
    return total_edits / reference_characters


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    nemo_root = arguments.nemo_root.resolve()
    model_path = arguments.model.resolve()
    manifest_path = arguments.manifest.resolve()
    criteria_path = arguments.criteria.resolve()
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not all(path.exists() for path in (nemo_root, model_path, manifest_path, criteria_path)):
        raise FileNotFoundError("nemo-root, model, manifest, and criteria must exist")
    if arguments.end_of_stream_padding_ms < 0:
        raise ValueError("end-of-stream-padding-ms must not be negative")
    criteria = Gate3Criteria.model_validate_json(criteria_path.read_text(encoding="utf-8"))
    model_sha256 = sha256_file(model_path)
    manifest_sha256 = sha256_file(manifest_path)
    if model_sha256 != arguments.expected_model_sha256 or model_sha256 != criteria.model_sha256:
        raise ValueError("frozen Gate 3 model SHA-256 mismatch")
    if manifest_sha256 != criteria.benchmark_manifest_sha256:
        raise ValueError("frozen Gate 3 benchmark SHA-256 mismatch")
    rows = load_manifest(manifest_path)
    if len(rows) != criteria.expected_case_count:
        raise ValueError("Gate 3 case count does not match frozen criteria")

    sys.path.insert(0, str(nemo_root))
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Gate 3 batch evaluation requires CUDA")
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    runtime_config = _runtime_config(
        model_path,
        end_of_stream_padding_ms=arguments.end_of_stream_padding_ms,
    )
    resolved_config = json.dumps(
        runtime_config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    config_sha256 = hashlib.sha256(resolved_config.encode()).hexdigest()
    _write_json(output_dir / "resolved-config.json", runtime_config)
    descriptor = build_model_descriptor(
        model_sha256=model_sha256,
        config_sha256=config_sha256,
    )
    asr_model, model_load_ms = load_asr_model(model_path, torch)

    first_audio = Path(str(rows[0]["audio_filepath"])).resolve()
    warmup_audio = inspect_audio(first_audio)
    warmup_tracker = StreamingTranscriptSession(
        session_id="gate3-warmup",
        model=descriptor,
        stabilization_window=arguments.stabilization_window,
    )
    warmup_result = _stream_audio(
        asr_model=asr_model,
        audio_path=first_audio,
        audio_duration_ms=float(warmup_audio["duration_ms"]),
        tracker=warmup_tracker,
        torch=torch,
        end_of_stream_padding_ms=arguments.end_of_stream_padding_ms,
        pace_realtime=False,
        compare_offline=True,
    )
    if not warmup_result.events or not warmup_result.state_released:
        raise RuntimeError("Gate 3 warmup did not complete cleanly")
    del warmup_result
    gc.collect()
    torch.cuda.synchronize()
    memory_baseline_allocated = int(torch.cuda.memory_allocated())
    memory_baseline_reserved = int(torch.cuda.memory_reserved())

    trace_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    memory_allocated_after: list[int] = []
    memory_reserved_after: list[int] = []
    infrastructure_failures: list[str] = []
    for index, row in enumerate(rows, start=1):
        case_id = str(row["case_id"])
        audio_path = Path(str(row["audio_filepath"])).resolve()
        audio = inspect_audio(audio_path)
        if audio["sha256"] != row["audio_sha256"]:
            raise ValueError(f"audio hash mismatch for {case_id}")
        if abs(float(audio["duration_ms"]) / 1000 - float(row["duration_seconds"])) > 0.02:
            raise ValueError(f"audio duration mismatch for {case_id}")
        tracker = StreamingTranscriptSession(
            session_id=case_id,
            model=descriptor,
            stabilization_window=arguments.stabilization_window,
        )
        torch.cuda.reset_peak_memory_stats()
        try:
            result = _stream_audio(
                asr_model=asr_model,
                audio_path=audio_path,
                audio_duration_ms=float(audio["duration_ms"]),
                tracker=tracker,
                torch=torch,
                end_of_stream_padding_ms=arguments.end_of_stream_padding_ms,
                pace_realtime=True,
                compare_offline=True,
            )
            peak_memory = int(torch.cuda.max_memory_allocated())
            metrics = tracker.metrics(
                surface_final_transcript=result.offline_transcript,
                peak_device_memory_bytes=peak_memory,
            )
            endpoint = endpoint_probe(audio_path)
            final_latency_ms = metrics.final_latency_ms or 0.0
            trace_row = {
                "case_id": case_id,
                "utterance_id": row.get("utterance_id"),
                "speaker_id": row["speaker_id"],
                "audio_sha256": audio["sha256"],
                "audio_duration_ms": audio["duration_ms"],
                "stream_duration_ms": (
                    float(audio["duration_ms"]) + arguments.end_of_stream_padding_ms
                ),
                "streaming_transcript": metrics.final_transcript,
                "offline_surface_transcript": result.offline_transcript,
                "metrics": metrics.model_dump(mode="json"),
                "finalization_lag_ms": max(0.0, final_latency_ms - float(audio["duration_ms"])),
                "realtime_factor": final_latency_ms / float(audio["duration_ms"]),
                "adapter_call_counts": result.adapter_call_counts,
                "state_released": result.state_released,
                "endpoint": endpoint,
            }
            trace_rows.append(trace_row)
            event_rows.extend({"case_id": case_id, **event} for event in result.events)
        except Exception as error:  # keep the frozen batch auditable after a per-case failure
            infrastructure_failures.append(f"{case_id}:{type(error).__name__}:{error}")
        finally:
            gc.collect()
            torch.cuda.synchronize()
            memory_allocated_after.append(int(torch.cuda.memory_allocated()))
            memory_reserved_after.append(int(torch.cuda.memory_reserved()))
        print(f"gate3 batch {index}/{len(rows)} complete: {case_id}", flush=True)

    lifecycle_audio = first_audio
    cancellation_tracker = StreamingTranscriptSession(
        session_id="gate3-cancellation",
        model=descriptor,
        stabilization_window=arguments.stabilization_window,
    )
    cancellation_result = _stream_audio(
        asr_model=asr_model,
        audio_path=lifecycle_audio,
        audio_duration_ms=float(warmup_audio["duration_ms"]),
        tracker=cancellation_tracker,
        torch=torch,
        end_of_stream_padding_ms=arguments.end_of_stream_padding_ms,
        cancel_after_chunks=2,
    )
    cancellation_correct = bool(
        cancellation_result.cancelled
        and cancellation_tracker.status.value == "cancelled"
        and cancellation_result.state_released
        and all(event["partial"] for event in cancellation_result.events)
    )

    reset_tracker = StreamingTranscriptSession(
        session_id="gate3-reset",
        model=descriptor,
        stabilization_window=arguments.stabilization_window,
    )
    reset_first = _stream_audio(
        asr_model=asr_model,
        audio_path=lifecycle_audio,
        audio_duration_ms=float(warmup_audio["duration_ms"]),
        tracker=reset_tracker,
        torch=torch,
        end_of_stream_padding_ms=arguments.end_of_stream_padding_ms,
    )
    reset_first_final = reset_tracker.metrics().final_transcript
    reset_tracker.reset()
    reset_second = _stream_audio(
        asr_model=asr_model,
        audio_path=lifecycle_audio,
        audio_duration_ms=float(warmup_audio["duration_ms"]),
        tracker=reset_tracker,
        torch=torch,
        end_of_stream_padding_ms=arguments.end_of_stream_padding_ms,
    )
    reset_second_metrics = reset_tracker.metrics()
    reset_correct = bool(
        reset_first.state_released
        and reset_second.state_released
        and reset_tracker.generation == 1
        and reset_second_metrics.final_transcript == reset_first_final
        and reset_second_metrics.final_transcript
    )

    complete = [row for row in trace_rows if row["metrics"]["final_transcript"]]
    endpoint_rows = [row["endpoint"] for row in trace_rows]
    endpoint_true_positive = sum(row["correct"] for row in endpoint_rows)
    endpoint_false_positive = sum(
        row["detected_ms"] is not None and not row["correct"] for row in endpoint_rows
    )
    endpoint_false_negative = len(endpoint_rows) - endpoint_true_positive
    endpoint_precision = (
        endpoint_true_positive / (endpoint_true_positive + endpoint_false_positive)
        if endpoint_true_positive + endpoint_false_positive
        else 0.0
    )
    endpoint_recall = (
        endpoint_true_positive / (endpoint_true_positive + endpoint_false_negative)
        if endpoint_true_positive + endpoint_false_negative
        else 0.0
    )
    endpoint_f1 = (
        2 * endpoint_precision * endpoint_recall / (endpoint_precision + endpoint_recall)
        if endpoint_precision + endpoint_recall
        else 0.0
    )
    endpoint_delays = [
        float(row["delay_ms"])
        for row in endpoint_rows
        if row["correct"] and row["delay_ms"] is not None
    ]
    first_partial_values = [
        float(row["metrics"]["first_partial_latency_ms"])
        for row in trace_rows
        if row["metrics"]["first_partial_latency_ms"] is not None
    ]
    chunk_latencies = [
        float(event["inference_latency_ms"])
        for event in event_rows
        if event["partial"] or not event["partial"]
    ]
    transcript_pairs = [
        (str(row["offline_surface_transcript"] or ""), str(row["streaming_transcript"]))
        for row in trace_rows
    ]
    all_adapter_names = {name for row in trace_rows for name in row["adapter_call_counts"].keys()}
    every_adapter_called = bool(all_adapter_names) and all(
        row["adapter_call_counts"].get(name, 0) > 0
        for row in trace_rows
        for name in all_adapter_names
    )
    evidence = Gate3Evidence(
        criteria_id=criteria.criteria_id,
        model_sha256=model_sha256,
        benchmark_manifest_sha256=manifest_sha256,
        case_count=len(rows),
        unique_case_count=len({row["case_id"] for row in rows}),
        speaker_count=len({row["speaker_id"] for row in rows}),
        complete_trace_count=len(complete),
        empty_final_count=len(trace_rows) - len(complete),
        nonempty_partial_rate=(
            sum(
                any(
                    event["case_id"] == row["case_id"] and event["partial"] and event["transcript"]
                    for event in event_rows
                )
                for row in trace_rows
            )
            / len(rows)
        ),
        mean_partial_stability=(
            fmean(float(row["metrics"]["partial_stability"]) for row in trace_rows)
            if trace_rows
            else 0.0
        ),
        trace_retraction_rate=(
            sum(row["metrics"]["stable_prefix_violations"] > 0 for row in trace_rows) / len(rows)
        ),
        exact_surface_agreement_rate=(
            sum(reference == hypothesis for reference, hypothesis in transcript_pairs) / len(rows)
        ),
        aggregate_surface_cer=_aggregate_cer(transcript_pairs),
        warm_first_partial_p95_ms=(
            percentile(first_partial_values, 0.95) if first_partial_values else 1_000_000.0
        ),
        finalization_lag_p95_ms=(
            percentile([float(row["finalization_lag_ms"]) for row in trace_rows], 0.95)
            if trace_rows
            else 1_000_000.0
        ),
        post_warmup_chunk_inference_p95_ms=(
            percentile(chunk_latencies, 0.95) if chunk_latencies else 1_000_000.0
        ),
        trace_realtime_factor_p95=(
            percentile([float(row["realtime_factor"]) for row in trace_rows], 0.95)
            if trace_rows
            else 1_000_000.0
        ),
        synthetic_endpoint_precision=endpoint_precision,
        synthetic_endpoint_recall=endpoint_recall,
        synthetic_endpoint_f1=endpoint_f1,
        endpoint_early_trigger_rate=(
            sum(row["early"] for row in endpoint_rows) / len(rows) if rows else 1.0
        ),
        endpoint_delay_p95_ms=(
            percentile(endpoint_delays, 0.95) if endpoint_delays else 1_000_000.0
        ),
        cancellation_correctness=float(cancellation_correct),
        reset_correctness=float(reset_correct),
        allocated_memory_growth_bytes=max(
            0,
            max(memory_allocated_after, default=memory_baseline_allocated)
            - memory_baseline_allocated,
        ),
        reserved_memory_growth_bytes=max(
            0,
            max(memory_reserved_after, default=memory_baseline_reserved) - memory_baseline_reserved,
        ),
        adapter_module_count=len(all_adapter_names),
        every_adapter_called=every_adapter_called,
        every_session_state_released=all(row["state_released"] for row in trace_rows),
        confidence_policy=CONFIDENCE_POLICY,
        infrastructure_failures=tuple(infrastructure_failures),
    )
    assessment = assess_gate3(criteria, evidence)
    _write_jsonl(output_dir / "traces.jsonl", trace_rows)
    _write_jsonl(output_dir / "events.jsonl", event_rows)
    _write_json(output_dir / "evidence.json", evidence.model_dump(mode="json"))
    _write_json(output_dir / "assessment.json", assessment.model_dump(mode="json"))
    summary = {
        "schema_version": "1.0.0",
        "decision": assessment.decision.value,
        "model_load_ms": model_load_ms,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "nemo": __import__("nemo").__version__,
            "device": torch.cuda.get_device_name(0),
        },
        "benchmark_manifest_sha256": manifest_sha256,
        "criteria_sha256": sha256_file(criteria_path),
        "evidence": evidence.model_dump(mode="json"),
        "assessment": assessment.model_dump(mode="json"),
        "lifecycle": {
            "cancellation_correct": cancellation_correct,
            "reset_correct": reset_correct,
        },
        "memory": {
            "baseline_allocated_bytes": memory_baseline_allocated,
            "baseline_reserved_bytes": memory_baseline_reserved,
            "allocated_after_each_session_bytes": memory_allocated_after,
            "reserved_after_each_session_bytes": memory_reserved_after,
        },
        "artifacts": {
            "events": "events.jsonl",
            "traces": "traces.jsonl",
            "evidence": "evidence.json",
            "assessment": "assessment.json",
            "resolved_config": "resolved-config.json",
        },
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    summary = run(build_parser().parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
