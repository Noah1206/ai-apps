"""Offline adapter for evaluating reproducible, precomputed model outputs."""

from __future__ import annotations

from pathlib import Path

from busan_lab.adapters.base import SurfaceASRAdapter
from busan_lab.schemas.asr import (
    ModelDescriptor,
    PrecomputedPrediction,
    SurfaceASRResult,
)
from busan_lab.schemas.benchmark import BenchmarkManifest


class PrecomputedSurfaceASRAdapter(SurfaceASRAdapter):
    """Read predictions exported by one fixed Surface ASR experiment."""

    def __init__(
        self,
        prediction_path: Path,
        manifest: BenchmarkManifest,
    ) -> None:
        predictions = [
            PrecomputedPrediction.model_validate_json(line)
            for line in prediction_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not predictions:
            raise ValueError("prediction JSONL is empty")
        models = {prediction.result.model for prediction in predictions}
        if len(models) != 1:
            raise ValueError("one prediction file must contain exactly one model version")
        self._model = models.pop()
        experiment_ids = {prediction.experiment_id for prediction in predictions}
        if len(experiment_ids) != 1:
            raise ValueError("one prediction file must contain exactly one experiment ID")
        self._experiment_id = experiment_ids.pop()
        for prediction in predictions:
            if (
                prediction.benchmark_id is not None
                and prediction.benchmark_id != manifest.benchmark_id
            ):
                raise ValueError("prediction benchmark ID does not match the manifest")
            if (
                prediction.benchmark_version is not None
                and prediction.benchmark_version != manifest.benchmark_version
            ):
                raise ValueError("prediction benchmark version does not match the manifest")
        self._by_audio_sha = {
            prediction.audio_sha256: prediction for prediction in predictions
        }
        if len(self._by_audio_sha) != len(predictions):
            raise ValueError("prediction JSONL contains duplicate audio hashes")
        expected = {
            (str(entry.utterance_id), entry.derived_audio_sha256)
            for entry in manifest.entries
        }
        actual = {
            (prediction.utterance_id, prediction.audio_sha256)
            for prediction in predictions
        }
        if actual != expected:
            raise ValueError("prediction JSONL must exactly match the benchmark entries")

    @property
    def model(self) -> ModelDescriptor:
        return self._model

    @property
    def experiment_id(self) -> str:
        return self._experiment_id

    def transcribe(
        self,
        audio_path: Path,
        *,
        utterance_id: str,
        audio_sha256: str,
    ) -> SurfaceASRResult:
        del audio_path
        try:
            prediction = self._by_audio_sha[audio_sha256]
        except KeyError as error:
            raise LookupError(f"no prediction for audio {audio_sha256}") from error
        if prediction.utterance_id != utterance_id:
            raise LookupError(
                f"prediction for audio {audio_sha256} belongs to "
                f"{prediction.utterance_id}, not {utterance_id}"
            )
        return prediction.result
