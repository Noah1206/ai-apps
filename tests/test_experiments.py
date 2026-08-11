import json
import zipfile
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from busan_lab.adapters.precomputed import PrecomputedSurfaceASRAdapter
from busan_lab.api import create_app
from busan_lab.cli import export_benchmark_bundle
from busan_lab.evaluation.reporting import finalize_human_reviewed_report
from busan_lab.evaluation.runner import BenchmarkRunner
from busan_lab.schemas.asr import PrecomputedPrediction
from busan_lab.schemas.experiment import HumanReview, ReviewVerdict
from busan_lab.storage import LabStorage
from tests.test_api import upload_utterance


def create_prediction(
    client: TestClient,
    *,
    utterance_id: str,
    experiment_id: str,
    model_version: str,
    hypothesis: str,
) -> dict[str, object]:
    response = client.post(
        "/api/predictions",
        json={
            "utterance_id": utterance_id,
            "experiment_id": experiment_id,
            "hypothesis_surface_text": hypothesis,
            "confidence": 0.94,
            "latency_ms": 120,
            "model": {
                "name": "surface-asr-fixture",
                "version": model_version,
            },
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_predictions_reviews_and_ab_comparison_persist(tmp_path: Path) -> None:
    data_root = tmp_path / "lab"
    client = TestClient(create_app(data_root))
    record = upload_utterance(client)
    utterance_id = str(record["utterance_id"])

    prediction_a = create_prediction(
        client,
        utterance_id=utterance_id,
        experiment_id="exp-baseline-a",
        model_version="v0",
        hypothesis="국밥 하나 주세요",
    )
    prediction_b = create_prediction(
        client,
        utterance_id=utterance_id,
        experiment_id="exp-baseline-b",
        model_version="v1",
        hypothesis="국밥 하나 주이소",
    )

    predictions = client.get(
        "/api/predictions",
        params={"utterance_id": utterance_id},
    )
    assert predictions.status_code == 200
    assert len(predictions.json()) == 2
    assert {item["experiment_id"] for item in predictions.json()} == {
        "exp-baseline-a",
        "exp-baseline-b",
    }

    comparison = client.post(
        "/api/comparisons",
        json={
            "prediction_a_id": prediction_a["prediction_id"],
            "prediction_b_id": prediction_b["prediction_id"],
        },
    )
    assert comparison.status_code == 200, comparison.text
    comparison_payload = comparison.json()
    assert comparison_payload["cer_delta_b_minus_a"] < 0
    assert comparison_payload["preservation_delta_b_minus_a"] == 1
    assert comparison_payload["overcorrection_delta_b_minus_a"] == -1

    queue = client.get("/api/review-queue")
    assert queue.status_code == 200
    assert [item["prediction_id"] for item in queue.json()] == [prediction_a["prediction_id"]]

    review = client.post(
        "/api/reviews",
        json={
            "prediction_id": prediction_a["prediction_id"],
            "reviewer_id": "reviewer-001",
            "verdict": "confirmed",
            "confirmed_failure_types": ["LANGUAGE_MODEL_BIAS"],
            "notes": "원음에서 주이소가 명확하게 들림",
        },
    )
    assert review.status_code == 201, review.text
    assert review.json()["utterance_id"] == utterance_id

    assert client.get("/api/review-queue").json() == []
    stored_reviews = client.get(
        "/api/reviews",
        params={"prediction_id": prediction_a["prediction_id"]},
    )
    assert stored_reviews.status_code == 200
    assert stored_reviews.json()[0]["verdict"] == "confirmed"

    reloaded_client = TestClient(create_app(data_root))
    reloaded_predictions = reloaded_client.get(
        "/api/predictions",
        params={"utterance_id": utterance_id},
    )
    assert len(reloaded_predictions.json()) == 2
    assert len(reloaded_client.get("/api/experiments").json()) == 2
    assert len(list((data_root / "predictions").glob("*.json"))) == 2
    assert len(list((data_root / "reviews").glob("*.json"))) == 1


def test_experiment_id_rejects_different_model_conditions(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "lab"))
    record = upload_utterance(client)
    utterance_id = str(record["utterance_id"])
    create_prediction(
        client,
        utterance_id=utterance_id,
        experiment_id="exp-fixed",
        model_version="v0",
        hypothesis="국밥 하나 주세요",
    )

    conflict = client.post(
        "/api/predictions",
        json={
            "utterance_id": utterance_id,
            "experiment_id": "exp-fixed",
            "hypothesis_surface_text": "국밥 하나 주이소",
            "confidence": 0.9,
            "model": {
                "name": "surface-asr-fixture",
                "version": "different-model",
            },
        },
    )

    assert conflict.status_code == 409
    assert "different model" in conflict.json()["detail"]
    assert len(client.get("/api/predictions").json()) == 1


def test_comparison_rejects_predictions_from_different_utterances(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "lab"))
    first = upload_utterance(client)
    second = upload_utterance(client)
    prediction_a = create_prediction(
        client,
        utterance_id=str(first["utterance_id"]),
        experiment_id="exp-shared",
        model_version="v0",
        hypothesis="국밥 하나 주세요",
    )
    prediction_b = create_prediction(
        client,
        utterance_id=str(second["utterance_id"]),
        experiment_id="exp-shared",
        model_version="v0",
        hypothesis="국밥 하나 주이소",
    )

    comparison = client.post(
        "/api/comparisons",
        json={
            "prediction_a_id": prediction_a["prediction_id"],
            "prediction_b_id": prediction_b["prediction_id"],
        },
    )

    assert comparison.status_code == 409
    assert "same utterance" in comparison.json()["detail"]


def test_review_rejects_unknown_prediction(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "lab"))
    response = client.post(
        "/api/reviews",
        json={
            "prediction_id": "00000000-0000-0000-0000-000000000000",
            "reviewer_id": "reviewer",
            "verdict": "uncertain",
        },
    )

    assert response.status_code == 404


def test_benchmark_runner_persists_experiment_and_predictions(tmp_path: Path) -> None:
    data_root = tmp_path / "lab"
    client = TestClient(create_app(data_root))
    record = upload_utterance(client)
    utterance_id = str(record["utterance_id"])
    benchmark_response = client.post(
        "/api/benchmarks",
        json={
            "benchmark_id": "runner-benchmark",
            "benchmark_version": "0.1.0",
            "utterance_ids": [utterance_id],
            "split": "test",
        },
    )
    assert benchmark_response.status_code == 201

    storage = LabStorage(data_root)
    manifest = storage.load_manifest("runner-benchmark", "0.1.0")
    bundle_path = export_benchmark_bundle(
        storage,
        manifest,
        tmp_path / "runner-benchmark--0.1.0.zip",
    )
    with zipfile.ZipFile(bundle_path) as bundle:
        assert set(bundle.namelist()) == {
            "benchmark.json",
            manifest.entries[0].derived_audio_path,
        }

    prediction_path = tmp_path / "predictions.jsonl"
    prediction_path.write_text(
        json.dumps(
            {
                "experiment_id": "task-002-fixture",
                "benchmark_id": "runner-benchmark",
                "benchmark_version": "0.1.0",
                "utterance_id": utterance_id,
                "audio_sha256": record["audio"]["derived"]["sha256"],
                "device": "fixture-cpu",
                "inference_timestamp": "2026-07-29T00:00:00Z",
                "result": {
                    "surface_text": "국밥 하나 주세요",
                    "confidence": None,
                    "confidence_supported": False,
                    "latency_ms": 120,
                    "model": {
                        "name": "nvidia-korean-conformer-ctc-fixture",
                        "version": "pretrained-v0",
                        "model_provider": "NVIDIA",
                        "model_family": "Conformer-CTC",
                        "decoder_type": "CTC greedy",
                        "fine_tuned": False,
                        "checkpoint_identifier": (
                            "nvidia/tao/speechtotext_ko_kr_conformer:deployable_v1.0"
                        ),
                        "checkpoint": "fixture://checkpoint",
                    },
                    "segments": [],
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = BenchmarkRunner(storage).run(
        manifest,
        PrecomputedSurfaceASRAdapter(prediction_path, manifest),
    )

    assert report.experiment_id == "task-002-fixture"
    assert report.metrics.utterance_count == 1
    assert report.metrics.cer > 0
    assert report.metrics.dialect_preservation_rate == 0
    assert report.metrics.context_overcorrection_rate == 1
    assert report.metrics.high_confidence_wrong_count == 0
    assert report.cases[0].confidence is None
    assert report.model.model_provider == "NVIDIA"
    assert report.model.model_family == "Conformer-CTC"
    assert report.model.decoder_type == "CTC greedy"
    assert report.model.target_language is None
    assert report.model.fine_tuned is False
    assert len(storage.list_experiments()) == 1
    stored_predictions = storage.list_predictions()
    assert len(stored_predictions) == 1
    assert stored_predictions[0].source.value == "precomputed"
    assert stored_predictions[0].experiment_id == report.experiment_id

    with pytest.raises(ValueError, match="missing human review"):
        finalize_human_reviewed_report(storage, report.experiment_id)

    prediction = stored_predictions[0]
    uncertain = HumanReview(
        prediction_id=prediction.prediction_id,
        utterance_id=prediction.utterance_id,
        reviewer_id="reviewer",
        verdict=ReviewVerdict.UNCERTAIN,
        created_at=report.created_at + timedelta(seconds=1),
    )
    storage.save_review(uncertain)
    with pytest.raises(ValueError, match="explanatory note"):
        finalize_human_reviewed_report(storage, report.experiment_id)

    invalid = HumanReview(
        review_id=uuid4(),
        prediction_id=prediction.prediction_id,
        utterance_id=prediction.utterance_id,
        reviewer_id="reviewer",
        verdict=ReviewVerdict.CONFIRMED,
        confirmed_failure_types=("MODLE",),
        notes="fixture typo",
        created_at=report.created_at + timedelta(seconds=2),
    )
    storage.save_review(invalid)
    with pytest.raises(ValueError, match="unknown confirmed failure type"):
        finalize_human_reviewed_report(storage, report.experiment_id)

    corrected = HumanReview(
        review_id=uuid4(),
        prediction_id=prediction.prediction_id,
        utterance_id=prediction.utterance_id,
        reviewer_id="reviewer",
        verdict=ReviewVerdict.CONFIRMED,
        confirmed_failure_types=("MODEL",),
        notes="원음과 다른 음절로 인식함",
        created_at=report.created_at + timedelta(seconds=3),
    )
    storage.save_review(corrected)
    reviewed, json_path, markdown_path = finalize_human_reviewed_report(
        storage,
        report.experiment_id,
    )

    assert reviewed.source_report_id == report.report_id
    assert reviewed.automatic_metrics == report.metrics
    assert reviewed.human_review.reviewed_prediction_count == 1
    assert reviewed.human_review.review_revision_count == 3
    assert reviewed.human_review.confirmed_count == 1
    assert reviewed.human_review.confirmed_failure_type_counts[0].failure_type == "MODEL"
    assert reviewed.cases[0].review_id == corrected.review_id
    assert json_path.is_file()
    assert markdown_path.is_file()
    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown.startswith("# TASK-002 Human-reviewed Baseline Report")
    assert "원음과 다른 음절로 인식함" in markdown


def test_rnnt_prediction_import_uses_final_surface_text_only(tmp_path: Path) -> None:
    data_root = tmp_path / "lab"
    client = TestClient(create_app(data_root))
    record = upload_utterance(client)
    utterance_id = str(record["utterance_id"])
    response = client.post(
        "/api/benchmarks",
        json={
            "benchmark_id": "rnnt-benchmark",
            "benchmark_version": "1.0.0",
            "utterance_ids": [utterance_id],
            "split": "test",
        },
    )
    assert response.status_code == 201

    payload = {
        "experiment_id": "task-003b-rnnt-fixture",
        "benchmark_id": "rnnt-benchmark",
        "benchmark_version": "1.0.0",
        "utterance_id": utterance_id,
        "audio_sha256": record["audio"]["derived"]["sha256"],
        "device": "fixture-nvidia-gpu",
        "inference_timestamp": "2026-07-30T00:00:00Z",
        "result": {
            "surface_text": "국밥 하나 주이소",
            "confidence": None,
            "confidence_supported": False,
            "latency_ms": 80,
            "model": {
                "name": "nvidia/nemotron-3.5-asr-streaming-0.6b",
                "version": "fixture-revision",
                "model_provider": "NVIDIA",
                "model_family": "FastConformer-RNNT",
                "decoder_type": "RNNT",
                "target_language": "ko-KR",
                "fine_tuned": False,
            },
            "segments": [],
        },
    }
    parsed = PrecomputedPrediction.model_validate(payload)
    assert parsed.result.surface_text == "국밥 하나 주이소"
    assert parsed.result.confidence is None
    assert parsed.result.confidence_supported is False
    assert parsed.result.segments == ()

    invalid_payload = json.loads(json.dumps(payload))
    invalid_payload["result"]["model"]["rnnt_decoder_state"] = {}
    with pytest.raises(ValidationError, match="Extra inputs"):
        PrecomputedPrediction.model_validate(invalid_payload)

    prediction_path = tmp_path / "rnnt-predictions.jsonl"
    prediction_path.write_text(
        json.dumps(payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    storage = LabStorage(data_root)
    manifest = storage.load_manifest("rnnt-benchmark", "1.0.0")
    report = BenchmarkRunner(storage).run(
        manifest,
        PrecomputedSurfaceASRAdapter(prediction_path, manifest),
    )

    assert report.metrics.utterance_count == 1
    assert report.cases[0].hypothesis_surface_text == "국밥 하나 주이소"
    assert report.cases[0].confidence is None
    assert report.model.model_family == "FastConformer-RNNT"
    assert report.model.decoder_type == "RNNT"
    assert report.model.target_language == "ko-KR"
    assert report.model.fine_tuned is False

    prediction = storage.list_predictions()[0]
    storage.save_review(
        HumanReview(
            prediction_id=prediction.prediction_id,
            utterance_id=prediction.utterance_id,
            reviewer_id="reviewer",
            verdict=ReviewVerdict.REJECTED,
        )
    )
    reviewed, _, markdown_path = finalize_human_reviewed_report(
        storage,
        report.experiment_id or "",
    )
    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown.startswith("# TASK-003B Human-reviewed Baseline Report")
    assert "Riva runtime" not in markdown
    assert any("호환 모델 런타임" in item for item in reviewed.limitations)
