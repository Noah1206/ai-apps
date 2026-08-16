import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

from fastapi.testclient import TestClient

from busan_lab.api import create_app
from busan_lab.schemas.common import ConsentRecord, DatasetSplit
from busan_lab.schemas.training_import import (
    TrainingRecordingImportEntry,
    TrainingRecordingImportManifest,
)
from tests.helpers import make_wav_bytes


def upload_utterance(client: TestClient) -> dict[str, object]:
    response = client.post(
        "/api/utterances",
        files={"file": ("busan-source.wav", make_wav_bytes(), "audio/wav")},
        data={
            "speaker_id": "busan-speaker-001",
            "region": "Busan",
            "device": "test-device",
            "environment": "quiet",
            "surface_text": "국밥 하나 주이소",
            "normalized_meaning": "국밥 하나 주세요",
            "dialect_expressions": json.dumps(
                [
                    {
                        "surface_form": "주이소",
                        "normalized_forms": ["주세요"],
                        "status": "candidate",
                    }
                ],
                ensure_ascii=False,
            ),
            "storage_allowed": "true",
            "research_use_allowed": "true",
            "model_training_allowed": "false",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def save_training_review_import(
    client: TestClient,
    record_payload: dict[str, object],
) -> TrainingRecordingImportManifest:
    storage = client.app.state.storage
    utterance_id = str(record_payload["utterance_id"])
    record = storage.load_utterance(utterance_id)
    consent = ConsentRecord(
        storage_allowed=True,
        research_use_allowed=True,
        model_training_allowed=True,
    )
    imported_record = record.model_copy(
        update={
            "source": "import",
            "dataset_split": DatasetSplit.TRAIN,
            "consent": consent,
        }
    )
    storage.save_utterance(imported_record)
    entry = TrainingRecordingImportEntry(
        prompt_id="T004-S001",
        source_filename="New Recording.m4a",
        source_audio_sha256=imported_record.audio.original.sha256,
        candidate_surface_text=imported_record.ground_truth.surface_text,
        utterance_id=imported_record.utterance_id,
        duration_ms=imported_record.audio.derived.duration_ms,
        audio_quality_passed=True,
    )
    manifest = TrainingRecordingImportManifest(
        import_id="task-004-api-review-v0",
        source_directory_name="recordings",
        prompt_sheet_sha256="0" * 64,
        speaker_id=imported_record.speaker.speaker_id,
        region=imported_record.speaker.region,
        device=imported_record.speaker.device,
        recording_environment=imported_record.speaker.environment,
        consent=consent,
        entries=(entry,),
    )
    storage.save_training_recording_import(manifest)
    return manifest


def test_quoted_labels_are_stored_as_individual_expressions(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "lab"))
    response = client.post(
        "/api/utterances",
        files={"file": ("quoted.wav", make_wav_bytes(), "audio/wav")},
        data={
            "speaker_id": "quoted-speaker",
            "surface_text": "와따 맛있노",
            "dialect_expressions": json.dumps(
                [{
                    "surface_form": '"와따", "맛있노"',
                    "normalized_forms": ['\'와"', '"맛있다"'],
                }],
                ensure_ascii=False,
            ),
            "storage_allowed": "true",
        },
    )

    assert response.status_code == 201, response.text
    record = response.json()
    expressions = record["ground_truth"]["dialect_expressions"]
    assert [
        (item["surface_form"], item["normalized_forms"]) for item in expressions
    ] == [("와따", ["와"]), ("맛있노", ["맛있다"])]

    updated = client.patch(
        f"/api/utterances/{record['utterance_id']}/labels",
        json={
            "surface_text": "니 지금 어데고?",
            "normalized_meaning": "너 지금 어디야?",
            "dialect_expressions": [{
                "surface_form": '"니", "어데고"',
                "normalized_forms": ['"너"', '"어디야\''],
            }],
            "changed_by": "labeler-001",
        },
    )

    assert updated.status_code == 200, updated.text
    expressions = updated.json()["ground_truth"]["dialect_expressions"]
    assert [
        (item["surface_form"], item["normalized_forms"]) for item in expressions
    ] == [("니", ["너"]), ("어데고", ["어디야"])]


def test_offline_vertical_slice(tmp_path: Path) -> None:
    data_root = tmp_path / "lab"
    client = TestClient(create_app(data_root))
    record = upload_utterance(client)
    utterance_id = record["utterance_id"]
    audio = record["audio"]
    assert audio["master"]["sample_rate_hz"] == 48_000
    assert audio["master"]["channels"] == 2
    assert {asset["role"] for asset in audio["derivatives"]} == {
        "asr_16k_mono",
        "pronunciation_24k_mono",
        "tts_48k_mono",
    }
    for role in (
        "original",
        "master_48k_stereo",
        "asr_16k_mono",
        "pronunciation_24k_mono",
        "tts_48k_mono",
    ):
        audio_response = client.get(f"/api/utterances/{utterance_id}/audio/{role}")
        assert audio_response.status_code == 200

    analysis = client.get(f"/api/utterances/{utterance_id}/analysis")
    assert analysis.status_code == 200
    assert analysis.json()["sample_rate_hz"] == 16_000
    assert analysis.json()["waveform_max"]
    assert analysis.json()["mel_db"]
    assert analysis.json()["f0_hz"]

    evaluation = client.post(
        "/api/evaluations",
        json={
            "utterance_id": utterance_id,
            "hypothesis_surface_text": "국밥 하나 주세요",
            "confidence": 0.94,
            "model": {
                "name": "surface-asr-fixture",
                "version": "pretrained-v0",
            },
        },
    )
    assert evaluation.status_code == 200
    result = evaluation.json()
    assert result["dialect"]["overcorrection_rate"] == 1
    assert result["high_confidence_wrong"] is True
    error_export = data_root / "exports" / "normalization-errors.jsonl"
    assert error_export.is_file()
    assert "LANGUAGE_MODEL_BIAS" in error_export.read_text(encoding="utf-8")

    benchmark = client.post(
        "/api/benchmarks",
        json={
            "benchmark_id": "busan-surface-v0",
            "benchmark_version": "0.1.0",
            "utterance_ids": [utterance_id],
            "split": "test",
        },
    )
    assert benchmark.status_code == 201, benchmark.text
    manifest = benchmark.json()
    assert manifest["frozen"] is True
    assert manifest["entries"][0]["surface_text"] == "국밥 하나 주이소"
    assert manifest["entries"][0]["normalized_meaning"] == "국밥 하나 주세요"


def test_persistent_upload_requires_storage_consent(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "lab"))
    response = client.post(
        "/api/utterances",
        files={"file": ("source.wav", make_wav_bytes(), "audio/wav")},
        data={
            "speaker_id": "speaker",
            "surface_text": "주이소",
            "storage_allowed": "false",
        },
    )

    assert response.status_code == 422
    assert "storage consent" in response.json()["detail"]


def test_invalid_audio_does_not_create_a_record(tmp_path: Path) -> None:
    data_root = tmp_path / "lab"
    client = TestClient(create_app(data_root))
    response = client.post(
        "/api/utterances",
        files={"file": ("not-audio.txt", b"hello", "text/plain")},
        data={
            "speaker_id": "speaker",
            "surface_text": "주이소",
            "storage_allowed": "true",
        },
    )

    assert response.status_code == 422
    assert list((data_root / "records").glob("*.json")) == []


def test_blocking_analysis_does_not_block_label_update(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "lab"))
    record = upload_utterance(client)
    utterance_id = str(record["utterance_id"])
    analysis_started = Event()
    release_analysis = Event()

    def blocking_analysis(_path: Path) -> dict[str, object]:
        analysis_started.set()
        assert release_analysis.wait(timeout=5)
        return {}

    monkeypatch.setattr("busan_lab.api.analyze_audio", blocking_analysis)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending_analysis = executor.submit(
            client.get,
            f"/api/utterances/{utterance_id}/analysis",
        )
        assert analysis_started.wait(timeout=5)
        response = client.patch(
            f"/api/utterances/{utterance_id}/labels",
            json={
                "surface_text": "국밥 하나 주이소예",
                "normalized_meaning": "국밥 하나 주세요",
                "dialect_expressions": [],
                "changed_by": "labeler-001",
            },
        )
        release_analysis.set()
        assert pending_analysis.result(timeout=5).status_code == 200

    assert response.status_code == 200, response.text
    assert response.json()["ground_truth"]["label_version"] == "label_v1"


def test_cloud_only_audio_returns_without_blocking_other_apis(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "lab"))
    record = upload_utterance(client)
    utterance_id = str(record["utterance_id"])
    monkeypatch.setattr("busan_lab.api.file_is_dataless", lambda _path: True)

    analysis = client.get(f"/api/utterances/{utterance_id}/analysis")
    audio = client.get(f"/api/utterances/{utterance_id}/audio/asr_16k_mono")
    health = client.get("/api/health")

    assert analysis.status_code == 503
    assert audio.status_code == 503
    assert "cloud-only" in analysis.json()["detail"]
    assert health.status_code == 200


def test_cloud_only_record_json_returns_503_instead_of_blocking(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Regression: evicted record JSON must fail fast, not hang the worker forever."""

    client = TestClient(create_app(tmp_path / "lab"))
    upload_utterance(client)
    monkeypatch.setattr("busan_lab.storage.file_is_dataless", lambda _path: True)

    listing = client.get("/api/utterances")
    health = client.get("/api/health")

    assert listing.status_code == 503
    assert "cloud-only" in listing.json()["detail"]
    assert health.status_code == 200


def test_research_ui_exposes_prediction_review_and_comparison_controls(
    tmp_path: Path,
) -> None:
    client = TestClient(create_app(tmp_path / "lab"))

    response = client.get("/")

    assert response.status_code == 200
    html = response.text
    assert "저장된 전체 발화 고정" in html
    for element_id in (
        'id="utterance-select"',
        'id="experiment-id"',
        'id="prediction-list"',
        'id="comparison-form"',
        'id="review-form"',
        'id="label-edit-form"',
        'id="delete-utterance"',
        'id="benchmark-list"',
        'id="training-review"',
        'id="training-review-item-select"',
        'id="training-review-approve"',
        'id="training-review-rerecord-button"',
    ):
        assert element_id in html

    script = client.get("/static/app.js")
    assert script.status_code == 200
    assert "utterance_ids: state.records.map" in script.text
    assert "buildDialectLabels" in script.text
    assert '"와따", "맛있노"' in script.text
    assert "/diagnostics" in script.text
    assert "관찰 오류" in script.text
    assert "(빈 출력)" in script.text


def test_task004_review_queue_preserves_revisions_and_import_ledger(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "lab"
    client = TestClient(create_app(data_root))
    record = upload_utterance(client)
    manifest = save_training_review_import(client, record)

    imports = client.get("/api/training-imports")
    assert imports.status_code == 200
    assert imports.json() == [
        {
            "task_id": "TASK-004",
            "import_id": manifest.import_id,
            "created_at": manifest.created_at.isoformat().replace("+00:00", "Z"),
            "speaker_id": manifest.speaker_id,
            "entry_count": 1,
        }
    ]

    queue_url = f"/api/training-imports/{manifest.import_id}/review-queue"
    queue = client.get(queue_url)
    assert queue.status_code == 200, queue.text
    assert queue.json()["candidate_count"] == 1
    assert queue.json()["reviewed_count"] == 0
    assert queue.json()["items"][0]["prompt_id"] == "T004-S001"

    approved = client.post(
        f"{queue_url}/T004-S001/decision",
        json={
            "reviewer_id": "labeler-001",
            "decision": "approve",
            "notes": "원음과 수집 문장을 대조함",
        },
    )
    assert approved.status_code == 200, approved.text
    assert approved.json()["ground_truth"]["label_status"] == "approved"
    assert approved.json()["ground_truth"]["label_version"] == "label_v1"
    after_approval = client.get(queue_url).json()
    assert after_approval["reviewed_count"] == 1
    assert after_approval["approved_count"] == 1

    rerecord = client.post(
        f"{queue_url}/T004-S001/decision",
        json={
            "reviewer_id": "labeler-001",
            "decision": "rerecord",
            "notes": "문장 앞부분이 잘림",
        },
    )
    assert rerecord.status_code == 200, rerecord.text
    assert rerecord.json()["ground_truth"]["label_status"] == "deprecated"
    assert rerecord.json()["ground_truth"]["label_version"] == "label_v2"
    after_rerecord = client.get(queue_url).json()
    assert after_rerecord["rerecord_count"] == 1
    assert after_rerecord["approved_count"] == 0
    revisions = client.get(
        f"/api/utterances/{record['utterance_id']}/label-revisions"
    )
    assert len(revisions.json()) == 2
    assert client.get(queue_url).json()["total_count"] == 1

    delete = client.delete(f"/api/utterances/{record['utterance_id']}")
    assert delete.status_code == 409
    assert "training import" in delete.json()["detail"]


def test_task004_review_queue_rejects_unknown_items_and_fields(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "lab"))
    record = upload_utterance(client)
    manifest = save_training_review_import(client, record)
    queue_url = f"/api/training-imports/{manifest.import_id}/review-queue"

    missing = client.post(
        f"{queue_url}/T004-S999/decision",
        json={"reviewer_id": "labeler-001", "decision": "approve"},
    )
    assert missing.status_code == 404

    unknown_field = client.post(
        f"{queue_url}/T004-S001/decision",
        json={
            "reviewer_id": "labeler-001",
            "decision": "approve",
            "unexpected": True,
        },
    )
    assert unknown_field.status_code == 422


def test_prediction_diagnostics_show_errors_without_mutating_prediction(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "lab"
    client = TestClient(create_app(data_root))
    record = upload_utterance(client)
    prediction = client.post(
        "/api/predictions",
        json={
            "utterance_id": record["utterance_id"],
            "experiment_id": "diagnostics-read-only",
            "hypothesis_surface_text": "",
            "confidence": 0.5,
            "model": {
                "name": "surface-asr-fixture",
                "version": "v0",
            },
        },
    )
    assert prediction.status_code == 201, prediction.text
    stored = prediction.json()
    assert stored["automatic_failure_candidates"] == []

    diagnostics = client.get(
        f"/api/predictions/{stored['prediction_id']}/diagnostics"
    )

    assert diagnostics.status_code == 200, diagnostics.text
    payload = diagnostics.json()
    assert payload["calibration_revision"] is None
    observed_errors = {item["observed_error"] for item in payload["observations"]}
    assert "DIALECT_EXPRESSION_LOST" in observed_errors
    assert "WORD_OR_SYLLABLE_OMISSION" in observed_errors
    assert payload["automatic_failure_candidates"] == ["MODEL"]
    reloaded = client.get(f"/api/predictions/{stored['prediction_id']}")
    assert reloaded.json()["automatic_failure_candidates"] == []


def test_label_correction_preserves_revision_and_audio(tmp_path: Path) -> None:
    data_root = tmp_path / "lab"
    client = TestClient(create_app(data_root))
    original = upload_utterance(client)
    utterance_id = str(original["utterance_id"])

    response = client.patch(
        f"/api/utterances/{utterance_id}/labels",
        json={
            "surface_text": "국밥 하나 주이소예",
            "normalized_meaning": "국밥 하나 주세요",
            "dialect_expressions": [
                {
                    "surface_form": "주이소예",
                    "normalized_forms": ["주세요", "주십시오"],
                    "status": "candidate",
                }
            ],
            "changed_by": "labeler-001",
            "reason": "원음을 다시 듣고 종결 표현을 수정",
        },
    )

    assert response.status_code == 200, response.text
    updated = response.json()
    assert updated["ground_truth"]["surface_text"] == "국밥 하나 주이소예"
    assert updated["ground_truth"]["label_version"] == "label_v1"
    assert updated["audio"]["original"]["sha256"] == original["audio"]["original"]["sha256"]

    revisions = client.get(f"/api/utterances/{utterance_id}/label-revisions")
    assert revisions.status_code == 200
    assert revisions.json()[0]["previous"]["surface_text"] == "국밥 하나 주이소"
    assert revisions.json()[0]["updated"]["surface_text"] == "국밥 하나 주이소예"
    assert len(list((data_root / "label_revisions" / utterance_id).glob("*.json"))) == 1


def test_frozen_benchmark_rejects_label_correction(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "lab"))
    original = upload_utterance(client)
    utterance_id = str(original["utterance_id"])
    benchmark = client.post(
        "/api/benchmarks",
        json={
            "benchmark_id": "frozen-label-test",
            "benchmark_version": "0.1.0",
            "utterance_ids": [utterance_id],
            "split": "test",
        },
    )
    assert benchmark.status_code == 201

    response = client.patch(
        f"/api/utterances/{utterance_id}/labels",
        json={
            "surface_text": "수정 시도",
            "normalized_meaning": None,
            "dialect_expressions": [],
            "changed_by": "labeler-001",
        },
    )

    assert response.status_code == 409
    assert "frozen benchmark" in response.json()["detail"]


def test_delete_benchmark_archives_manifest_and_unlocks_labels(tmp_path: Path) -> None:
    data_root = tmp_path / "lab"
    client = TestClient(create_app(data_root))
    record = upload_utterance(client)
    utterance_id = str(record["utterance_id"])
    benchmark = client.post(
        "/api/benchmarks",
        json={
            "benchmark_id": "removable-benchmark",
            "benchmark_version": "0.1.0",
            "utterance_ids": [utterance_id],
            "split": "test",
        },
    )
    assert benchmark.status_code == 201

    response = client.delete("/api/benchmarks/removable-benchmark/0.1.0")

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["entry_count"] == 1
    assert (data_root / result["archived_path"]).is_file()
    assert client.get("/api/benchmarks").json() == []
    label_update = client.patch(
        f"/api/utterances/{utterance_id}/labels",
        json={
            "surface_text": "국밥 하나 주이소예",
            "normalized_meaning": "국밥 하나 주세요",
            "dialect_expressions": [],
            "changed_by": "labeler-001",
        },
    )
    assert label_update.status_code == 200


def test_delete_unknown_benchmark_returns_not_found(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "lab"))

    response = client.delete("/api/benchmarks/missing/0.1.0")

    assert response.status_code == 404


def test_delete_utterance_moves_record_and_unshared_audio_to_trash(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "lab"
    client = TestClient(create_app(data_root))
    record = upload_utterance(client)
    utterance_id = str(record["utterance_id"])
    asset_paths = {
        record["audio"]["original"]["relative_path"],
        record["audio"]["master"]["relative_path"],
        *(
            asset["relative_path"]
            for asset in record["audio"]["derivatives"]
        ),
    }

    response = client.delete(f"/api/utterances/{utterance_id}")

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["utterance_id"] == utterance_id
    assert client.get(f"/api/utterances/{utterance_id}").status_code == 404
    assert not (data_root / "records" / f"{utterance_id}.json").exists()
    assert all(not (data_root / relative_path).exists() for relative_path in asset_paths)
    archive_root = data_root / "trash" / utterance_id / result["archive_id"]
    assert (archive_root / "records" / f"{utterance_id}.json").is_file()
    assert all((archive_root / relative_path).is_file() for relative_path in asset_paths)


def test_delete_preserves_audio_shared_by_another_record(tmp_path: Path) -> None:
    data_root = tmp_path / "lab"
    client = TestClient(create_app(data_root))
    first = upload_utterance(client)
    second = upload_utterance(client)

    response = client.delete(f"/api/utterances/{first['utterance_id']}")

    assert response.status_code == 200, response.text
    assert response.json()["preserved_shared_paths"]
    second_audio = client.get(
        f"/api/utterances/{second['utterance_id']}/audio/asr_16k_mono"
    )
    assert second_audio.status_code == 200


def test_delete_archives_label_prediction_review_and_error_export(tmp_path: Path) -> None:
    data_root = tmp_path / "lab"
    client = TestClient(create_app(data_root))
    record = upload_utterance(client)
    utterance_id = str(record["utterance_id"])
    label_update = client.patch(
        f"/api/utterances/{utterance_id}/labels",
        json={
            "surface_text": "국밥 하나 주이소예",
            "normalized_meaning": "국밥 하나 주세요",
            "dialect_expressions": [
                {
                    "surface_form": "주이소예",
                    "normalized_forms": ["주세요"],
                    "status": "candidate",
                }
            ],
            "changed_by": "labeler-001",
        },
    )
    assert label_update.status_code == 200
    prediction = client.post(
        "/api/predictions",
        json={
            "utterance_id": utterance_id,
            "experiment_id": "delete-evidence-test",
            "hypothesis_surface_text": "국밥 하나 주세요",
            "confidence": 0.95,
            "model": {
                "name": "surface-asr-fixture",
                "version": "v0",
            },
        },
    )
    assert prediction.status_code == 201
    prediction_id = prediction.json()["prediction_id"]
    review = client.post(
        "/api/reviews",
        json={
            "prediction_id": prediction_id,
            "reviewer_id": "reviewer-001",
            "verdict": "confirmed",
        },
    )
    assert review.status_code == 201

    response = client.delete(f"/api/utterances/{utterance_id}")

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["removed_export_rows"] == 1
    assert list((data_root / "label_revisions" / utterance_id).glob("*.json")) == []
    assert list((data_root / "predictions").glob("*.json")) == []
    assert list((data_root / "reviews").glob("*.json")) == []
    export_path = data_root / "exports" / "normalization-errors.jsonl"
    assert export_path.is_file()
    assert utterance_id not in export_path.read_text(encoding="utf-8")
    archive_root = data_root / "trash" / utterance_id / result["archive_id"]
    assert list((archive_root / "label_revisions" / utterance_id).glob("*.json"))
    assert list((archive_root / "predictions").glob("*.json"))
    assert list((archive_root / "reviews").glob("*.json"))


def test_frozen_benchmark_rejects_utterance_deletion(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "lab"))
    record = upload_utterance(client)
    utterance_id = str(record["utterance_id"])
    benchmark = client.post(
        "/api/benchmarks",
        json={
            "benchmark_id": "frozen-delete-test",
            "benchmark_version": "0.1.0",
            "utterance_ids": [utterance_id],
            "split": "test",
        },
    )
    assert benchmark.status_code == 201

    response = client.delete(f"/api/utterances/{utterance_id}")

    assert response.status_code == 409
    assert client.get(f"/api/utterances/{utterance_id}").status_code == 200
