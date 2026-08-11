"""Thin model boundary for official Transformers Nemotron offline inference."""

from __future__ import annotations

import hashlib
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from contract import ContractError


@dataclass(frozen=True, slots=True)
class TranscriptionTrace:
    """JSON-safe evidence from raw generation through Adapter output."""

    processor_call: dict[str, Any]
    processor_inputs: dict[str, Any]
    model_generate_call: dict[str, Any]
    raw_model_output: dict[str, Any]
    decoded_with_special_tokens: str | None
    special_decode_error: dict[str, str] | None
    decoded_transcript: str | None
    batch_decoded_transcripts: list[str] | None
    batch_decode_error: dict[str, str] | None
    adapter_transcript: str | None
    adapter_transformation: str
    extraction_error: dict[str, str] | None


class SpeechModelAdapter(Protocol):
    model_id: str
    requested_revision: str
    resolved_revision: str
    model_artifact_sha256: str
    model_cache_path: str
    model_load_ms: float
    device_name: str
    generation_output_type: str | None

    def warmup(self) -> float: ...

    def synchronize(self) -> None: ...

    def transcribe(self, audio_path: Path) -> str: ...

    def transcribe_with_trace(self, audio_path: Path) -> TranscriptionTrace: ...


class TransformersNemotronAdapter:
    """Load the fixed public checkpoint through its documented RNNT AutoModel API."""

    expected_model_class = "Nemotron3_5AsrForRNNT"
    expected_processor_class = "Nemotron3_5AsrProcessor"
    expected_sample_rate = 16_000

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        target_language: str,
        cache_dir: Path | None,
    ) -> None:
        started = time.perf_counter()
        self._require_full_commit_sha(revision)
        try:
            import numpy as np
            import torch
            from huggingface_hub import HfApi, hf_hub_download
            from transformers import AutoModelForRNNT, AutoProcessor
        except ImportError as error:
            raise RuntimeError(
                "GPU runtime dependencies are missing; install requirements.txt "
                "inside the documented NVIDIA environment"
            ) from error

        if not torch.cuda.is_available():
            raise RuntimeError("Nemotron inference requires an available NVIDIA CUDA GPU")

        resolved_revision = HfApi().model_info(model_id, revision=revision).sha
        if resolved_revision != revision:
            raise RuntimeError(
                f"requested revision {revision} resolved to unexpected {resolved_revision}"
            )

        cache_value = str(cache_dir.expanduser().resolve()) if cache_dir is not None else None
        artifact_path = Path(
            hf_hub_download(
                repo_id=model_id,
                filename="model.safetensors",
                revision=resolved_revision,
                cache_dir=cache_value,
            )
        )
        artifact_hash = _sha256_file(artifact_path)

        processor = AutoProcessor.from_pretrained(
            model_id,
            revision=resolved_revision,
            cache_dir=cache_value,
        )
        model = AutoModelForRNNT.from_pretrained(
            model_id,
            revision=resolved_revision,
            cache_dir=cache_value,
        )
        model = model.to("cuda")
        if type(processor).__name__ != self.expected_processor_class:
            raise RuntimeError(
                f"unexpected processor class: {type(processor).__name__}; "
                f"expected {self.expected_processor_class}"
            )
        if type(model).__name__ != self.expected_model_class:
            raise RuntimeError(
                f"unexpected model class: {type(model).__name__}; "
                f"expected {self.expected_model_class}"
            )
        sample_rate = int(processor.feature_extractor.sampling_rate)
        if sample_rate != self.expected_sample_rate:
            raise RuntimeError(f"unexpected model sample rate: {sample_rate}")
        if int(model.config.max_symbols_per_step) != 10:
            raise RuntimeError(
                f"unexpected RNNT max_symbols_per_step: {model.config.max_symbols_per_step}"
            )
        model.eval()
        model_device = model.device
        if model_device.type != "cuda":
            raise RuntimeError(f"model was not placed on CUDA: {model_device}")

        self.model_id = model_id
        self.requested_revision = revision
        self.resolved_revision = resolved_revision
        self.model_artifact_sha256 = artifact_hash
        self.model_cache_path = _sanitize_cache_path(artifact_path, resolved_revision)
        self.model_load_ms = (time.perf_counter() - started) * 1000
        self.device_name = torch.cuda.get_device_name(model_device)
        self.generation_output_type: str | None = None
        self._target_language = target_language
        self._model = model
        self._processor = processor
        self._np = np
        self._torch = torch

    def synchronize(self) -> None:
        self._torch.cuda.synchronize(self._model.device)

    def warmup(self) -> float:
        """Run one synthetic one-second input, never a benchmark utterance."""

        audio = self._np.zeros(self.expected_sample_rate, dtype=self._np.float32)
        self.synchronize()
        started = time.perf_counter()
        trace = self._transcribe_array_with_trace(audio)
        self.synchronize()
        self._require_adapter_transcript(trace)
        return (time.perf_counter() - started) * 1000

    def transcribe(self, audio_path: Path) -> str:
        return self._require_adapter_transcript(self.transcribe_with_trace(audio_path))

    def transcribe_with_trace(self, audio_path: Path) -> TranscriptionTrace:
        return self._transcribe_array_with_trace(_read_pcm16_mono(audio_path, self._np))

    def _transcribe_array_with_trace(self, audio: Any) -> TranscriptionTrace:
        inputs = self._processor(
            audio,
            sampling_rate=self.expected_sample_rate,
            language=self._target_language,
            return_tensors="pt",
        )
        processor_inputs = _summarize_mapping(inputs)
        inputs = inputs.to(self._model.device, dtype=self._model.dtype)
        model_generate_call = {
            "api": "model.generate",
            "return_dict_in_generate": True,
            "streamer_passed": False,
            "streaming_input_generator_passed": False,
            "is_streaming_passed_to_processor": False,
            "input_keys": sorted(str(key) for key in inputs.keys()),
        }
        with self._torch.inference_mode():
            output = self._model.generate(**inputs, return_dict_in_generate=True)
        self.generation_output_type = type(output).__name__
        raw_model_output = _summarize_generation_output(output)
        decoded_with_special_tokens: str | None = None
        special_decode_error: dict[str, str] | None = None
        decoded_transcript: str | None = None
        batch_decoded_transcripts: list[str] | None = None
        batch_decode_error: dict[str, str] | None = None
        adapter_transcript: str | None = None
        extraction_error: dict[str, str] | None = None
        if hasattr(output, "sequences"):
            try:
                decoded_special = self._processor.decode(
                    output.sequences,
                    skip_special_tokens=False,
                )
                if not isinstance(decoded_special, str):
                    raise RuntimeError(
                        "unexpected special-token decode type: "
                        f"{type(decoded_special).__name__}"
                    )
                decoded_with_special_tokens = decoded_special
            except Exception as error:
                special_decode_error = {
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
        try:
            if not hasattr(output, "sequences"):
                raise RuntimeError(
                    f"unexpected RNNT generation output {type(output).__name__}: "
                    "missing sequences"
                )
            decoded = self._processor.decode(
                output.sequences,
                skip_special_tokens=True,
            )
            if not isinstance(decoded, str):
                raise RuntimeError(
                    f"unexpected decoded transcript type: {type(decoded).__name__}"
                )
            decoded_transcript = decoded
            adapter_transcript = decoded
        except Exception as error:
            extraction_error = {
                "error_type": type(error).__name__,
                "message": str(error),
            }
        if hasattr(output, "sequences") and hasattr(self._processor, "batch_decode"):
            try:
                batch_decoded = self._processor.batch_decode(
                    output.sequences,
                    skip_special_tokens=True,
                )
                if isinstance(batch_decoded, (list, tuple)) and all(
                    isinstance(item, str) for item in batch_decoded
                ):
                    batch_decoded_transcripts = list(batch_decoded)
            except Exception as error:
                batch_decode_error = {
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
        return TranscriptionTrace(
            processor_call={
                "sampling_rate": self.expected_sample_rate,
                "language": self._target_language,
                "return_tensors": "pt",
                "is_streaming_passed": False,
                "is_first_audio_chunk_passed": False,
            },
            processor_inputs=processor_inputs,
            model_generate_call=model_generate_call,
            raw_model_output=raw_model_output,
            decoded_with_special_tokens=decoded_with_special_tokens,
            special_decode_error=special_decode_error,
            decoded_transcript=decoded_transcript,
            batch_decoded_transcripts=batch_decoded_transcripts,
            batch_decode_error=batch_decode_error,
            adapter_transcript=adapter_transcript,
            adapter_transformation="identity_no_postprocessing",
            extraction_error=extraction_error,
        )

    @staticmethod
    def _require_adapter_transcript(trace: TranscriptionTrace) -> str:
        if trace.adapter_transcript is not None:
            return trace.adapter_transcript
        error = trace.extraction_error or {
            "error_type": "RuntimeError",
            "message": "Adapter did not produce a transcript",
        }
        raise RuntimeError(f"{error['error_type']}: {error['message']}")

    @staticmethod
    def _require_full_commit_sha(revision: str) -> None:
        if len(revision) != 40 or any(
            character not in "0123456789abcdef" for character in revision
        ):
            raise ContractError("model revision must be a full lowercase Hugging Face commit SHA")


def _read_pcm16_mono(audio_path: Path, np: Any) -> Any:
    try:
        with wave.open(str(audio_path), "rb") as audio:
            if (
                audio.getnchannels() != 1
                or audio.getframerate() != 16_000
                or audio.getsampwidth() != 2
                or audio.getcomptype() != "NONE"
            ):
                raise ContractError(f"unexpected WAV contract at inference time: {audio_path.name}")
            frames = audio.readframes(audio.getnframes())
    except wave.Error as error:
        raise ContractError(f"unreadable WAV at inference time: {audio_path.name}") from error
    if not frames:
        raise ContractError(f"empty WAV at inference time: {audio_path.name}")
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sanitize_cache_path(path: Path, revision: str) -> str:
    """Record a useful cache identifier without leaking the user's home path."""

    model_root = next(
        (index for index, part in enumerate(path.parts) if part.startswith("models--")),
        None,
    )
    if model_root is None:
        return f"<model-cache>/snapshot/{revision}/model.safetensors"
    return str(Path("<model-cache>", *path.parts[model_root:]))


def _summarize_mapping(value: Any) -> dict[str, Any]:
    if not hasattr(value, "keys"):
        return {"type": type(value).__name__, "keys": [], "fields": {}}
    keys = sorted(str(key) for key in value.keys())
    fields: dict[str, Any] = {}
    for key in keys:
        try:
            item = value[key]
        except (KeyError, TypeError):
            item = getattr(value, key, None)
        fields[key] = _summarize_value(item, include_values=False)
    return {
        "type": type(value).__name__,
        "keys": keys,
        "fields": fields,
    }


def _summarize_generation_output(output: Any) -> dict[str, Any]:
    keys = sorted(str(key) for key in output.keys()) if hasattr(output, "keys") else []
    if not keys and hasattr(output, "__dict__"):
        keys = sorted(
            str(key)
            for key, value in vars(output).items()
            if not key.startswith("_") and value is not None
        )
    fields: dict[str, Any] = {}
    for key in keys:
        try:
            value = output[key]
        except (KeyError, TypeError):
            value = getattr(output, key, None)
        fields[key] = _summarize_value(value, include_values=key == "sequences")
    if "sequences" not in fields and hasattr(output, "sequences"):
        fields["sequences"] = _summarize_value(output.sequences, include_values=True)
        keys = sorted((*keys, "sequences"))
    return {
        "type": type(output).__name__,
        "keys": keys,
        "fields": fields,
    }


def _summarize_value(value: Any, *, include_values: bool) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        summary: dict[str, Any] = {
            "type": type(value).__name__,
            "shape": [int(size) for size in value.shape],
            "dtype": str(value.dtype),
            "device": str(getattr(value, "device", "unknown")),
        }
        if include_values:
            try:
                summary["values"] = value.detach().cpu().tolist()
            except (AttributeError, RuntimeError, TypeError, ValueError) as error:
                summary["values_error"] = f"{type(error).__name__}: {error}"
        return summary
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "length": len(value),
            "items": [
                _summarize_value(item, include_values=False)
                for item in value[:8]
            ],
            "truncated": len(value) > 8,
        }
    if isinstance(value, dict):
        return {
            str(key): _summarize_value(item, include_values=False)
            for key, item in value.items()
        }
    return {"type": type(value).__name__}
