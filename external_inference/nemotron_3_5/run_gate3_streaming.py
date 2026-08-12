"""Run one cache-aware Gate 3 streaming trace with the frozen Gate 2 model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import wave
from pathlib import Path
from typing import Any, NamedTuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from busan_lab.schemas.asr import ModelDescriptor  # noqa: E402
from busan_lab.streaming import StreamingTranscriptSession  # noqa: E402

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
MODEL_REVISION = "f3d333391852ba876df169dcc9ba902d25b6ab0b"
TARGET_LANGUAGE = "ko-KR"
DEFAULT_ATT_CONTEXT = (56, 3)
DEFAULT_CHUNK_SIZE_MS = 320.0
DEFAULT_END_OF_STREAM_PADDING_MS = 320.0
DEFAULT_MODEL_SHA256 = "eb7d5112329504db72ffcb2c638e245e68f35801cda14a33682478d4a2ae85ee"


class StreamingGPUResult(NamedTuple):
    events: list[dict[str, Any]]
    offline_transcript: str | None
    adapter_call_counts: dict[str, int]
    state_released: bool
    cancelled: bool
    session_wall_ms: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--surface-predictions", type=Path, default=None)
    parser.add_argument("--expected-model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--stabilization-window", type=int, default=3)
    parser.add_argument(
        "--end-of-stream-padding-ms",
        type=float,
        default=DEFAULT_END_OF_STREAM_PADDING_MS,
    )
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_audio(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as audio:
        channels = audio.getnchannels()
        sample_width = audio.getsampwidth()
        sample_rate = audio.getframerate()
        frame_count = audio.getnframes()
        compression = audio.getcomptype()
    if (channels, sample_width, sample_rate, compression) != (1, 2, 16_000, "NONE"):
        raise ValueError("Gate 3 audio must be 16 kHz mono uncompressed PCM16 WAV")
    if frame_count <= 0:
        raise ValueError("Gate 3 audio must not be empty")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "channels": channels,
        "sample_width_bytes": sample_width,
        "sample_rate_hz": sample_rate,
        "frame_count": frame_count,
        "duration_ms": frame_count / sample_rate * 1000,
    }


def find_surface_prediction(path: Path | None, audio_sha256: str) -> str | None:
    if path is None:
        return None
    matches: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("audio_sha256") != audio_sha256:
                continue
            result = row.get("result")
            if not isinstance(result, dict) or not isinstance(result.get("surface_text"), str):
                raise ValueError(f"invalid surface prediction at line {line_number}")
            matches.append(result["surface_text"])
    if len(matches) > 1:
        raise ValueError("surface prediction file has duplicate audio_sha256 entries")
    return matches[0] if matches else None


def _runtime_config(
    model_path: Path,
    *,
    end_of_stream_padding_ms: float = DEFAULT_END_OF_STREAM_PADDING_MS,
) -> dict[str, Any]:
    return {
        "runtime_candidate": "runtime-v5",
        "implementation": "CacheAwareStreamingAudioBuffer+conformer_stream_step",
        "model_path": str(model_path),
        "device": "cuda:0",
        # NeMo's reference cache-aware runner explicitly rejects non-float32 compute.
        "compute_dtype": "float32",
        "amp": False,
        "decoder_type": "rnnt",
        "decoding_strategy": "greedy_batch",
        "attention_context": list(DEFAULT_ATT_CONTEXT),
        "chunk_size_ms": DEFAULT_CHUNK_SIZE_MS,
        "target_language": TARGET_LANGUAGE,
        "strip_language_tags": True,
        "online_normalization": False,
        "pad_and_drop_preencoded": False,
        "end_of_stream_padding_ms": end_of_stream_padding_ms,
        "end_of_stream_padding_policy": "zero_pcm_streaming_only_offline_unpadded",
        "end_of_stream_pacing_policy": "process_synthetic_flush_immediately_after_explicit_eof",
        "finalization_policy": "flush_zero_pcm_then_finalize_last_hypothesis",
        "tail_output_policy": "keep_all_outputs_on_last_yieldable_chunk",
    }


def _extract_transcription(hypotheses: Any) -> str:
    if not hypotheses:
        return ""
    hypothesis = hypotheses[0]
    text = hypothesis.text if hasattr(hypothesis, "text") else hypothesis
    if not isinstance(text, str):
        raise TypeError("NeMo returned a non-string streaming transcription")
    return text


def _streaming_buffer_has_next_chunk(streaming_buffer: Any) -> bool:
    """Mirror NeMo's iterator stop rule after the current chunk was yielded."""

    if streaming_buffer.buffer_idx >= streaming_buffer.buffer.size(-1):
        return False
    chunk_size = streaming_buffer.streaming_cfg.chunk_size
    if isinstance(chunk_size, list):
        chunk_size = chunk_size[1]
    remaining_frames = min(
        int(chunk_size),
        int(streaming_buffer.buffer.size(-1) - streaming_buffer.buffer_idx),
    )
    sampling_frames = streaming_buffer.sampling_frames
    if sampling_frames is None:
        return True
    if isinstance(sampling_frames, list):
        sampling_frames = sampling_frames[1]
    return remaining_frames >= int(sampling_frames)


def _read_pcm16_with_end_padding(path: Path, padding_ms: float) -> Any:
    """Read validated PCM16 audio and append deterministic zero-valued PCM."""

    if padding_ms < 0:
        raise ValueError("end-of-stream-padding-ms must not be negative")
    import numpy as np

    with wave.open(str(path), "rb") as audio:
        if (
            audio.getnchannels(),
            audio.getsampwidth(),
            audio.getframerate(),
            audio.getcomptype(),
        ) != (1, 2, 16_000, "NONE"):
            raise ValueError("Gate 3 audio must be 16 kHz mono uncompressed PCM16 WAV")
        frames = audio.readframes(audio.getnframes())
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    padding_samples = round(padding_ms * 16_000 / 1000)
    if padding_samples:
        samples = np.pad(samples, (0, padding_samples), mode="constant")
    return samples


def _realtime_pace_target_ms(*, audio_end_ms: float, source_duration_ms: float) -> float:
    """Return the capture time that must elapse before processing a chunk.

    Synthetic end padding exists only to flush cached model state after explicit
    EOF. It is already available at that point and therefore adds compute time,
    but no artificial capture wait.
    """

    if audio_end_ms < 0 or source_duration_ms <= 0:
        raise ValueError("audio timing must be positive")
    return min(audio_end_ms, source_duration_ms)


def _stream_audio(
    *,
    asr_model: Any,
    audio_path: Path,
    audio_duration_ms: float,
    tracker: StreamingTranscriptSession,
    torch: Any,
    end_of_stream_padding_ms: float = DEFAULT_END_OF_STREAM_PADDING_MS,
    pace_realtime: bool = False,
    compare_offline: bool = False,
    cancel_after_chunks: int | None = None,
) -> StreamingGPUResult:
    from nemo.collections.asr.parts.utils.streaming_utils import (
        CacheAwareStreamingAudioBuffer,
    )

    adapter_modules = [
        (name, module)
        for name, module in asr_model.named_modules()
        if module.__class__.__name__ == "LinearAdapter"
    ]
    adapter_call_counts = {name: 0 for name, _module in adapter_modules}

    def hook_for(name: str) -> Any:
        def count_adapter_call(_module: Any, _inputs: Any, _output: Any) -> None:
            adapter_call_counts[name] += 1

        return count_adapter_call

    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=asr_model,
        online_normalization=False,
        pad_and_drop_preencoded=False,
    )
    streaming_buffer.append_audio_file(str(audio_path), stream_id=-1)
    offline_transcript: str | None = None
    if compare_offline:
        with torch.inference_mode():
            processed_signal, processed_signal_length = streaming_buffer.get_all_audios()
            offline_outputs = asr_model.conformer_stream_step(
                processed_signal=processed_signal.to(torch.float32),
                processed_signal_length=processed_signal_length,
                return_transcription=True,
            )
        offline_transcript = _extract_transcription(offline_outputs[1])
        del offline_outputs, processed_signal, processed_signal_length

    # The offline comparator remains bound to the unmodified source audio. The
    # streaming path gets one explicit zero-PCM flush chunk so terminal tokens
    # are not silently stranded in NeMo's unyieldable residual feature tail.
    if end_of_stream_padding_ms:
        padded_audio = _read_pcm16_with_end_padding(audio_path, end_of_stream_padding_ms)
        streaming_buffer.reset_buffer()
        streaming_buffer.append_audio(padded_audio, stream_id=-1)
        del padded_audio
    stream_duration_ms = audio_duration_ms + end_of_stream_padding_ms

    hooks = [module.register_forward_hook(hook_for(name)) for name, module in adapter_modules]
    cache_last_channel, cache_last_time, cache_last_channel_len = (
        asr_model.encoder.get_initial_cache_state(batch_size=1)
    )
    previous_hypotheses = None
    pred_out_stream = None
    last_transcript = ""
    events: list[dict[str, Any]] = []
    cancelled = False
    session_started = time.perf_counter()
    try:
        for step_number, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer):
            final_chunk = not _streaming_buffer_has_next_chunk(streaming_buffer)
            audio_end_ms = (
                stream_duration_ms
                if final_chunk
                else min((step_number + 1) * DEFAULT_CHUNK_SIZE_MS, stream_duration_ms)
            )
            if pace_realtime:
                pace_target_ms = _realtime_pace_target_ms(
                    audio_end_ms=audio_end_ms,
                    source_duration_ms=audio_duration_ms,
                )
                target_time = session_started + pace_target_ms / 1000
                remaining_seconds = target_time - time.perf_counter()
                if remaining_seconds > 0:
                    time.sleep(remaining_seconds)
            step_started = time.perf_counter()
            with torch.inference_mode():
                (
                    pred_out_stream,
                    transcribed_texts,
                    cache_last_channel,
                    cache_last_time,
                    cache_last_channel_len,
                    previous_hypotheses,
                ) = asr_model.conformer_stream_step(
                    processed_signal=chunk_audio.to(torch.float32),
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=cache_last_channel,
                    cache_last_time=cache_last_time,
                    cache_last_channel_len=cache_last_channel_len,
                    keep_all_outputs=final_chunk,
                    previous_hypotheses=previous_hypotheses,
                    previous_pred_out=pred_out_stream,
                    drop_extra_pre_encoded=(
                        0
                        if step_number == 0
                        else asr_model.encoder.streaming_cfg.drop_extra_pre_encoded
                    ),
                    return_transcription=True,
                )
            torch.cuda.synchronize()
            step_finished = time.perf_counter()
            last_transcript = _extract_transcription(transcribed_texts)
            event = tracker.observe(
                last_transcript,
                audio_end_ms=audio_end_ms,
                emitted_at_ms=(step_finished - session_started) * 1000,
                inference_latency_ms=(step_finished - step_started) * 1000,
                # This smoke ends on explicit end-of-input; endpoint detection is
                # deliberately not credited until a dedicated endpoint corpus exists.
                endpoint_detected=False,
                final=False,
                confidence=None,
            )
            events.append(event.model_dump(mode="json"))
            if cancel_after_chunks is not None and step_number + 1 >= cancel_after_chunks:
                tracker.cancel()
                cancelled = True
                break
        if events and not cancelled:
            final_event = tracker.observe(
                last_transcript,
                audio_end_ms=stream_duration_ms,
                emitted_at_ms=(time.perf_counter() - session_started) * 1000,
                inference_latency_ms=0.0,
                endpoint_detected=False,
                final=True,
                confidence=None,
            )
            events.append(final_event.model_dump(mode="json"))
    finally:
        for hook in hooks:
            hook.remove()
        streaming_buffer.reset_buffer()
        cache_last_channel = None
        cache_last_time = None
        cache_last_channel_len = None
        previous_hypotheses = None
        pred_out_stream = None
    state_released = streaming_buffer.buffer is None and streaming_buffer.streams_length is None
    return StreamingGPUResult(
        events=events,
        offline_transcript=offline_transcript,
        adapter_call_counts=adapter_call_counts,
        state_released=state_released,
        cancelled=cancelled,
        session_wall_ms=(time.perf_counter() - session_started) * 1000,
    )


def load_asr_model(model_path: Path, torch: Any) -> tuple[Any, float]:
    from nemo.collections.asr.models import EncDecRNNTBPEModelWithPrompt
    from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig

    load_started = time.perf_counter()
    asr_model = EncDecRNNTBPEModelWithPrompt.restore_from(
        restore_path=str(model_path),
        map_location=torch.device("cuda:0"),
    )
    asr_model.encoder.set_default_att_context_size(att_context_size=list(DEFAULT_ATT_CONTEXT))
    decoding_config = RNNTDecodingConfig(fused_batch_size=-1)
    if hasattr(asr_model, "cur_decoder"):
        asr_model.change_decoding_strategy(decoding_config, decoder_type="rnnt")
    else:
        asr_model.change_decoding_strategy(decoding_config)
    asr_model.set_inference_prompt(TARGET_LANGUAGE)
    asr_model.decoding.set_strip_lang_tags(True)
    asr_model = asr_model.to(device=torch.device("cuda:0"), dtype=torch.float32)
    asr_model.eval()
    torch.cuda.synchronize()
    return asr_model, (time.perf_counter() - load_started) * 1000


def build_model_descriptor(*, model_sha256: str, config_sha256: str) -> ModelDescriptor:
    return ModelDescriptor(
        name=MODEL_ID,
        version=MODEL_REVISION,
        model_provider="NVIDIA",
        model_family="Cache-Aware FastConformer-RNNT",
        decoder_type="RNNT greedy_batch",
        target_language=TARGET_LANGUAGE,
        fine_tuned=True,
        checkpoint_identifier=f"nemo://final-gate2-selected.nemo#sha256:{model_sha256}",
        config_hash=config_sha256,
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    nemo_root = arguments.nemo_root.expanduser().resolve()
    model_path = arguments.model.expanduser().resolve()
    audio_path = arguments.audio.expanduser().resolve()
    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not nemo_root.is_dir() or not model_path.is_file() or not audio_path.is_file():
        raise FileNotFoundError("nemo-root, model, and audio must exist")
    if arguments.stabilization_window < 2:
        raise ValueError("stabilization-window must be at least 2")
    if arguments.end_of_stream_padding_ms < 0:
        raise ValueError("end-of-stream-padding-ms must not be negative")

    audio = inspect_audio(audio_path)
    model_sha256 = sha256_file(model_path)
    if arguments.expected_model_sha256 and model_sha256 != arguments.expected_model_sha256:
        raise ValueError("frozen Gate 3 model SHA-256 mismatch")
    surface_final = find_surface_prediction(arguments.surface_predictions, audio["sha256"])

    sys.path.insert(0, str(nemo_root))
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Gate 3 streaming smoke requires CUDA")
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
    config_sha256 = hashlib.sha256(resolved_config.encode("utf-8")).hexdigest()
    _write_json(output_dir / "resolved-config.json", runtime_config)

    model = build_model_descriptor(
        model_sha256=model_sha256,
        config_sha256=config_sha256,
    )
    session_id = arguments.session_id or f"gate3-smoke-{audio['sha256'][:12]}"
    tracker = StreamingTranscriptSession(
        session_id=session_id,
        model=model,
        stabilization_window=arguments.stabilization_window,
    )

    asr_model, model_load_ms = load_asr_model(model_path, torch)
    torch.cuda.reset_peak_memory_stats()
    streaming_result = _stream_audio(
        asr_model=asr_model,
        audio_path=audio_path,
        audio_duration_ms=audio["duration_ms"],
        tracker=tracker,
        torch=torch,
        end_of_stream_padding_ms=arguments.end_of_stream_padding_ms,
    )
    events = streaming_result.events
    adapter_module_count = len(streaming_result.adapter_call_counts)
    adapter_call_count = sum(streaming_result.adapter_call_counts.values())
    state_released = streaming_result.state_released
    peak_memory = int(torch.cuda.max_memory_allocated())
    metrics = tracker.metrics(
        surface_final_transcript=surface_final,
        peak_device_memory_bytes=peak_memory,
    )
    events_path = output_dir / "events.jsonl"
    events_path.write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events),
        encoding="utf-8",
    )
    failure_reasons = []
    if not events:
        failure_reasons.append("no_streaming_events")
    if not metrics.final_transcript:
        failure_reasons.append("empty_final_transcript")
    if not state_released:
        failure_reasons.append("session_state_not_released")
    if adapter_module_count == 0:
        failure_reasons.append("no_linear_adapters_found")
    elif adapter_call_count == 0:
        failure_reasons.append("linear_adapters_not_called")
    summary = {
        "schema_version": "1.0.0",
        "status": "smoke_failed" if failure_reasons else "smoke_passed",
        "failure_reasons": failure_reasons,
        "model": model.model_dump(mode="json"),
        "model_sha256": model_sha256,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "nemo": __import__("nemo").__version__,
            "device": torch.cuda.get_device_name(0),
            "model_load_ms": model_load_ms,
        },
        "streaming": {
            "implementation": runtime_config["implementation"],
            "encoder_cache_enabled": True,
            "feature_cache_enabled": True,
            "attention_context": list(DEFAULT_ATT_CONTEXT),
            "chunk_size_ms": DEFAULT_CHUNK_SIZE_MS,
            "source_audio_duration_ms": audio["duration_ms"],
            "stream_duration_ms": audio["duration_ms"] + arguments.end_of_stream_padding_ms,
            "end_of_stream_padding_ms": arguments.end_of_stream_padding_ms,
            "language_prompt": TARGET_LANGUAGE,
            "session_state_released": state_released,
            "adapter_module_count": adapter_module_count,
            "adapter_call_count": adapter_call_count,
            "endpoint_mode": "explicit_end_of_input_only",
            "endpoint_accuracy_tested": False,
            "cancellation_tested": False,
            "confidence_supported": False,
        },
        "audio": audio,
        "surface_final_transcript": surface_final,
        "metrics": metrics.model_dump(mode="json"),
        "artifacts": {
            "events": events_path.name,
            "resolved_config": "resolved-config.json",
            "resolved_config_sha256": config_sha256,
        },
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    summary = run(build_parser().parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "smoke_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
