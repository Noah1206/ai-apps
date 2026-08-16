"""Release-oriented offline speech runtimes.

The ASR backend deliberately reproduces the frozen Gate 2 inference path.  It
does not add context biasing, reference conditioning, or text post-processing.
Heavy NeMo imports are delayed until ``load`` so the API contract remains
testable on machines without the GPU environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import wave
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import BoundedSemaphore
from typing import Any, Protocol

import numpy as np

MODEL_CLASS = "nemo.collections.asr.models.rnnt_bpe_models_prompt.EncDecRNNTBPEModelWithPrompt"
TARGET_LANGUAGE = "ko-KR"
DEFAULT_MODEL_SHA256 = "eb7d5112329504db72ffcb2c638e245e68f35801cda14a33682478d4a2ae85ee"
DEFAULT_ADAPTER_SHA256 = "d0716aff0f05580d51dcad138c3036128d925d4821149842625f760ed0b7b954"
DEFAULT_MODEL_VERSION = "busan-asr-gate2-pass-20260812"
DEFAULT_MODEL_PATH = Path("data/lab/gate2/final-attempt-v2/model/final-gate2-selected.nemo")


class SpeechRuntimeError(RuntimeError):
    """An inference runtime is unavailable or failed safely."""


class SpeechQueueTimeout(SpeechRuntimeError):
    """The single-GPU inference slot was not acquired before its deadline."""


class AudioContractError(ValueError):
    """Uploaded audio does not satisfy the practice API contract."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ASRInference:
    transcript: str
    latency_ms: float


@dataclass(frozen=True, slots=True)
class TTSInference:
    audio_url: str
    cached: bool
    duration_ms: int
    sample_rate: int


class ASRBackend(Protocol):
    model_version: str

    @property
    def loaded(self) -> bool: ...

    @property
    def unavailable_reason(self) -> str | None: ...

    def load(self) -> None: ...

    def transcribe(self, audio_path: Path) -> ASRInference: ...


class TTSBackend(Protocol):
    model_version: str

    @property
    def loaded(self) -> bool: ...

    @property
    def unavailable_reason(self) -> str | None: ...

    def load(self) -> None: ...

    def generate(self, *, sentence_id: str, text: str, voice: str) -> TTSInference: ...


@dataclass(frozen=True, slots=True)
class PreparedAudio:
    path: Path
    duration_ms: float


class PracticeAudioPreprocessor:
    """Decode an upload into a temporary 16 kHz mono PCM16 WAV."""

    def __init__(
        self,
        *,
        ffmpeg_binary: str = "ffmpeg",
        ffprobe_binary: str = "ffprobe",
        min_duration_ms: float = 300,
        max_duration_ms: float = 15_000,
    ) -> None:
        self.ffmpeg_binary = ffmpeg_binary
        self.ffprobe_binary = ffprobe_binary
        self.min_duration_ms = min_duration_ms
        self.max_duration_ms = max_duration_ms

    def prepare(self, source: Path, destination: Path) -> PreparedAudio:
        if not source.is_file() or source.stat().st_size == 0:
            raise AudioContractError("UNSUPPORTED_AUDIO", "The upload is not decodable audio.")
        duration_ms = self._probe_duration_ms(source)
        if duration_ms < self.min_duration_ms:
            raise AudioContractError(
                "AUDIO_TOO_SHORT",
                "The recording is too short to analyze.",
            )
        if duration_ms > self.max_duration_ms:
            raise AudioContractError(
                "AUDIO_TOO_LONG",
                f"The recording is longer than the {self.max_duration_ms / 1000:g} second limit.",
            )

        command = [
            self.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-map_metadata",
            "-1",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise AudioContractError(
                "UPLOAD_FAILED", "The audio converter is unavailable."
            ) from error
        if completed.returncode != 0 or not destination.is_file():
            raise AudioContractError(
                "UNSUPPORTED_AUDIO", "The upload is not a supported audio stream."
            )

        with wave.open(str(destination), "rb") as audio:
            if (
                audio.getframerate() != 16_000
                or audio.getnchannels() != 1
                or audio.getsampwidth() != 2
            ):
                raise AudioContractError(
                    "UNSUPPORTED_AUDIO", "Audio conversion did not produce 16 kHz mono PCM16."
                )
            samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(
                np.float32
            )
        if samples.size == 0:
            raise AudioContractError("NO_SPEECH", "No speech was detected in the recording.")
        normalized = samples / 32768.0
        rms = float(np.sqrt(np.mean(np.square(normalized, dtype=np.float64))))
        silence_ratio = float(np.mean(np.abs(normalized) < 0.01))
        if rms < 0.001 or silence_ratio > 0.99:
            raise AudioContractError("NO_SPEECH", "No speech was detected in the recording.")
        return PreparedAudio(path=destination, duration_ms=duration_ms)

    def _probe_duration_ms(self, source: Path) -> float:
        command = [
            self.ffprobe_binary,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=duration",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(source),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise AudioContractError(
                "UPLOAD_FAILED", "The audio inspector is unavailable."
            ) from error
        if completed.returncode != 0:
            raise AudioContractError("UNSUPPORTED_AUDIO", "The upload is not decodable audio.")
        try:
            payload = json.loads(completed.stdout)
            streams = payload.get("streams", [])
            stream_duration = streams[0].get("duration") if streams else None
            duration = float(stream_duration or payload["format"]["duration"])
        except (IndexError, KeyError, TypeError, ValueError) as error:
            raise AudioContractError(
                "UNSUPPORTED_AUDIO", "The audio duration could not be determined."
            ) from error
        return duration * 1000.0


class UnavailableASRBackend:
    def __init__(self, reason: str, model_version: str = "unavailable") -> None:
        self.model_version = model_version
        self._reason = reason

    @property
    def loaded(self) -> bool:
        return False

    @property
    def unavailable_reason(self) -> str:
        return self._reason

    def load(self) -> None:
        raise SpeechRuntimeError(self._reason)

    def transcribe(self, audio_path: Path) -> ASRInference:
        del audio_path
        raise SpeechRuntimeError(self._reason)


class UnavailableTTSBackend:
    def __init__(self, reason: str, model_version: str = "unavailable") -> None:
        self.model_version = model_version
        self._reason = reason

    @property
    def loaded(self) -> bool:
        return False

    @property
    def unavailable_reason(self) -> str:
        return self._reason

    def load(self) -> None:
        raise SpeechRuntimeError(self._reason)

    def generate(self, *, sentence_id: str, text: str, voice: str) -> TTSInference:
        del sentence_id, text, voice
        raise SpeechRuntimeError(self._reason)


class NemoOfflineASRBackend:
    """Load and serve the hash-pinned Gate 2 NeMo artifact once per process."""

    def __init__(
        self,
        *,
        model_path: Path,
        expected_model_sha256: str,
        expected_adapter_sha256: str,
        model_version: str,
        device: str = "cuda:0",
        inference_timeout_seconds: float = 60,
    ) -> None:
        self.model_path = model_path.expanduser().resolve()
        self.expected_model_sha256 = expected_model_sha256.lower()
        self.expected_adapter_sha256 = expected_adapter_sha256.lower()
        self.model_version = model_version
        self.device = device
        self.inference_timeout_seconds = inference_timeout_seconds
        self._model: Any | None = None
        self._torch: Any | None = None
        self._reason: str | None = "model has not been loaded"
        self._semaphore = BoundedSemaphore(value=1)
        self._adapter_probe = {"registered_modules": 0, "forward_calls": 0}

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def unavailable_reason(self) -> str | None:
        return self._reason

    def load(self) -> None:
        if self.loaded:
            return
        if not self.model_path.is_file():
            self._reason = f"checkpoint not found: {self.model_path}"
            raise SpeechRuntimeError(self._reason)
        actual_hash = sha256_file(self.model_path)
        if actual_hash != self.expected_model_sha256:
            self._reason = f"checkpoint SHA-256 mismatch: {actual_hash}"
            raise SpeechRuntimeError(self._reason)

        try:
            import torch  # type: ignore[import-not-found]
            from nemo.collections.asr.models import ASRModel  # type: ignore[import-not-found]
            from omegaconf import OmegaConf, open_dict  # type: ignore[import-not-found]
        except ImportError as error:
            self._reason = f"NeMo runtime dependency is unavailable: {error}"
            raise SpeechRuntimeError(self._reason) from error
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self._reason = "CUDA is required by the frozen ASR release configuration"
            raise SpeechRuntimeError(self._reason)

        try:
            model = ASRModel.restore_from(
                str(self.model_path), map_location=torch.device("cpu"), strict=True
            )
            class_name = f"{type(model).__module__}.{type(model).__name__}"
            if class_name != MODEL_CLASS:
                raise SpeechRuntimeError(f"unexpected model class: {class_name}")
            if (
                str(model.cfg.decoding.strategy) != "greedy_batch"
                or int(model.cfg.decoding.greedy.max_symbols) != 10
            ):
                raise SpeechRuntimeError(
                    "checkpoint decoding config is not the frozen Gate 2 config"
                )

            adapter_items = [
                (name, tensor)
                for name, tensor in model.state_dict().items()
                if ".adapter_layer.busan_ko_kr_v0." in name
            ]
            if len(adapter_items) != 96 or sum(t.numel() for _, t in adapter_items) != 1_622_016:
                raise SpeechRuntimeError("fine-tuned adapter tensor count mismatch")
            adapter_hash = tensor_digest(adapter_items, uint8_dtype=torch.uint8)
            if adapter_hash != self.expected_adapter_sha256:
                raise SpeechRuntimeError(f"adapter state SHA-256 mismatch: {adapter_hash}")
            model.set_enabled_adapters(enabled=False)
            model.set_enabled_adapters("encoder:busan_ko_kr_v0", enabled=True)

            decoding = OmegaConf.create(OmegaConf.to_container(model.cfg.decoding, resolve=True))
            with open_dict(decoding):
                decoding.strategy = "greedy_batch"
                decoding.greedy.max_symbols = 10
                decoding.compute_timestamps = False
                decoding.preserve_alignments = False
            model.change_decoding_strategy(decoding, verbose=False)

            original_setup = model._setup_transcribe_dataloader

            def setup_reference_free_offline_dataloader(config: Any) -> Any:
                non_lhotse_config = deepcopy(config)
                non_lhotse_config["use_lhotse"] = False
                return original_setup(non_lhotse_config)

            model._setup_transcribe_dataloader = setup_reference_free_offline_dataloader
            for module_name, module in model.named_modules():
                if (
                    "adapter_layer.busan_ko_kr_v0" in module_name
                    and type(module).__name__ == "LinearAdapter"
                ):
                    self._adapter_probe["registered_modules"] += 1

                    def count_adapter_forward(
                        _module: Any,
                        _inputs: Any,
                        _output: Any,
                        probe: dict[str, int] = self._adapter_probe,
                    ) -> None:
                        probe["forward_calls"] += 1

                    module.register_forward_hook(count_adapter_forward)
            if self._adapter_probe["registered_modules"] != 24:
                raise SpeechRuntimeError(
                    "expected 24 executable adapter modules, got "
                    f"{self._adapter_probe['registered_modules']}"
                )
            model.freeze()
            model.eval()
            model = model.to(torch.device(self.device))
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
        except Exception as error:
            self._reason = str(error)
            if isinstance(error, SpeechRuntimeError):
                raise
            raise SpeechRuntimeError(f"ASR checkpoint loading failed: {error}") from error

        self._model = model
        self._torch = torch
        self._reason = None

    def transcribe(self, audio_path: Path) -> ASRInference:
        if self._model is None or self._torch is None:
            raise SpeechRuntimeError(self._reason or "ASR model is unavailable")
        acquired = self._semaphore.acquire(timeout=self.inference_timeout_seconds)
        if not acquired:
            raise SpeechQueueTimeout("ASR inference queue timed out")
        try:
            calls_before = self._adapter_probe["forward_calls"]
            if self.device.startswith("cuda"):
                self._torch.cuda.synchronize()
            started = time.perf_counter()
            raw = self._model.transcribe(
                audio=[str(audio_path)],
                batch_size=1,
                return_hypotheses=False,
                num_workers=0,
                verbose=False,
                target_lang=TARGET_LANGUAGE,
            )
            if self.device.startswith("cuda"):
                self._torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - started) * 1000.0
            if self._adapter_probe["forward_calls"] <= calls_before:
                raise SpeechRuntimeError("ASR inference did not execute the Busan adapter")
            return ASRInference(
                transcript=normalize_transcription(raw),
                latency_ms=latency_ms,
            )
        except SpeechRuntimeError:
            raise
        except Exception as error:
            raise SpeechRuntimeError(f"ASR inference failed: {error}") from error
        finally:
            self._semaphore.release()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(items: list[tuple[str, Any]], *, uint8_dtype: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(items):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(contiguous.view(uint8_dtype).numpy().tobytes())
    return digest.hexdigest()


def normalize_transcription(raw: Any) -> str:
    candidate = raw
    if isinstance(candidate, tuple):
        candidate = candidate[0]
    if isinstance(candidate, list):
        if len(candidate) != 1:
            raise SpeechRuntimeError(f"expected one transcription, received {len(candidate)}")
        candidate = candidate[0]
    if hasattr(candidate, "text"):
        candidate = candidate.text
    if not isinstance(candidate, str):
        raise SpeechRuntimeError(f"unsupported transcription result: {type(candidate)!r}")
    return candidate


def git_commit_from_environment() -> str:
    return os.getenv("BUSAN_GIT_COMMIT", "unknown")
