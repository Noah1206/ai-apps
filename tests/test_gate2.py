from __future__ import annotations

import hashlib
import json
import tarfile
import wave
from pathlib import Path
from typing import Any

from busan_lab.gate2 import (
    assess_gate2,
    canonical_benchmark_sha256,
    compare_gate2_predictions,
    prepare_blind_ab_review,
    run_blind_ab_review,
    validate_blind_ab_review,
    validate_evaluation_dataset,
    verify_recovery_bundle,
    verify_reproducibility_artifacts,
)
from busan_lab.schemas.benchmark import BenchmarkManifest
from busan_lab.schemas.gate2 import (
    EvaluationExclusionRegistry,
    Gate2Criteria,
    Gate2EvaluationEntry,
    Gate2EvaluationManifest,
    Gate2Evidence,
    ReproducibilityArtifact,
    ReproducibilitySpec,
)

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _criteria() -> Gate2Criteria:
    return Gate2Criteria(
        pilot_max_cer=0.35,
        pilot_min_dialect_preservation=0.6,
        human_min_fine_tuned_preference_rate=0.7,
        independent_min_utterances=2,
        independent_min_speakers=2,
        independent_min_relative_cer_improvement=0.1,
        independent_min_dialect_preservation_delta=0.15,
        standard_min_utterances=2,
        standard_min_speakers=2,
        standard_max_relative_cer_regression=0.15,
        standard_max_absolute_cer_regression=0.03,
        empty_output_increase_allowed=0,
    )


def test_benchmark_raw_serializations_have_one_semantic_identity() -> None:
    stored_path = ROOT / "data/lab/manifests/busan-surface-v0--1.0.0.json"
    exported_path = ROOT / "artifacts/task-005/evaluation-v0/benchmark/benchmark.json"
    stored = BenchmarkManifest.model_validate_json(stored_path.read_text(encoding="utf-8"))
    exported = BenchmarkManifest.model_validate_json(exported_path.read_text(encoding="utf-8"))

    assert _sha(stored_path) != _sha(exported_path)
    assert stored == exported
    assert canonical_benchmark_sha256(stored) == (
        "700d352edb4a4e9321b48ec6cd312bec6ad1d4c48fa2bedbcf80a2ca23a67f8c"
    )


def test_artifact_verifier_requires_presence_pinned_hash_and_match(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"task-005")
    base = {
        "experiment_id": "task-005-test",
        "base_model_id": "nvidia/nemotron-3.5-asr-streaming-0.6b",
        "base_model_revision": "a" * 40,
        "nemo_revision": "b" * 40,
        "train_dataset_sha256": "c" * 64,
        "validation_dataset_sha256": "d" * 64,
        "benchmark_package_sha256": "e" * 64,
        "checkpoint_selection_criterion": "lowest validation loss",
    }
    spec = ReproducibilitySpec(
        **base,
        artifacts=(
            ReproducibilityArtifact(
                artifact_id="artifact",
                relative_path="artifact.bin",
                expected_sha256=_sha(artifact),
                purpose="test",
            ),
        ),
    )
    assert verify_reproducibility_artifacts(tmp_path, spec)["passed"] is True

    unpinned = ReproducibilitySpec(
        **base,
        artifacts=(
            ReproducibilityArtifact(
                artifact_id="artifact",
                relative_path="artifact.bin",
                expected_sha256=None,
                purpose="test",
            ),
        ),
    )
    result = verify_reproducibility_artifacts(tmp_path, unpinned)
    assert result["passed"] is False
    assert "not pinned" in result["errors"][0]


def test_recovery_bundle_verifier_checks_every_indexed_file(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    artifact = bundle / "artifact.bin"
    artifact.write_bytes(b"checkpoint")
    manifest = {
        "files": [
            {
                "relative_path": artifact.name,
                "size_bytes": artifact.stat().st_size,
                "sha256": _sha(artifact),
            }
        ]
    }
    _write_json(bundle / "MANIFEST.json", manifest)
    (bundle / "MANIFEST.sha256").write_text(_sha(bundle / "MANIFEST.json"), encoding="utf-8")
    archive_path = tmp_path / "bundle.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(bundle, arcname="bundle")

    result = verify_recovery_bundle(bundle, source_archive=archive_path)
    assert result["passed"] is True
    assert result["verified_files"] == 1


def test_all_ten_blinded_items_are_reviewable_without_model_identity(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.json"
    results_path = tmp_path / "results.json"
    queue, results = prepare_blind_ab_review(
        source_queue_path=(
            ROOT / "artifacts/task-005/evaluation-v0/output/blinded-ab-review-queue.json"
        ),
        benchmark_path=ROOT / "artifacts/task-005/evaluation-v0/benchmark/benchmark.json",
        output_queue_path=queue_path,
        output_results_path=results_path,
    )
    _write_json(queue_path, queue)
    _write_json(results_path, results)

    assert len(queue["items"]) == len(results.items) == 10
    serialized = json.dumps(queue, ensure_ascii=False).casefold()
    assert all(
        identifier not in serialized
        for identifier in ("nemotron", "pretrained", "fine_tuned", "nvidia")
    )
    assert all((queue_path.parent / item["audio_path"]).is_file() for item in queue["items"])
    assert validate_blind_ab_review(queue_path=queue_path, results_path=results_path) == {
        "schema_version": "1.0.0",
        "valid": True,
        "complete": False,
        "total_items": 10,
        "reviewed_items": 0,
        "summary": None,
    }


def test_terminal_blind_review_saves_complete_structured_result(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    queue_path = tmp_path / "queue.json"
    results_path = tmp_path / "results.json"
    queue, results = prepare_blind_ab_review(
        source_queue_path=(
            ROOT / "artifacts/task-005/evaluation-v0/output/blinded-ab-review-queue.json"
        ),
        benchmark_path=ROOT / "artifacts/task-005/evaluation-v0/benchmark/benchmark.json",
        output_queue_path=queue_path,
        output_results_path=results_path,
    )
    queue["items"] = queue["items"][:1]
    results = results.model_copy(update={"items": results.items[:1]})
    _write_json(queue_path, queue)
    _write_json(results_path, results)
    answers = iter(("a", "b", "s", "n", "d"))
    played: list[Path] = []
    monkeypatch.setattr(
        "busan_lab.gate2._play_audio",
        lambda audio_path: played.append(audio_path),
    )

    completed = run_blind_ab_review(
        queue_path=queue_path,
        results_path=results_path,
        reviewer_id="reviewer-1",
        open_audio=True,
        input_fn=lambda _: next(answers),
    )

    assert completed.status == "complete"
    assert completed.items[0].transcript_preference == "A"
    assert completed.items[0].dialect_preservation_preference == "B"
    assert completed.items[0].meaning_distortion_a == "absent"
    assert completed.items[0].overcorrection_a == "present"
    assert completed.items[0].overcorrection_b == "present"
    assert played == [(queue_path.parent / queue["items"][0]["audio_path"]).resolve()]


def test_held_out_dataset_validation_checks_audio_and_overlap(tmp_path: Path) -> None:
    entries: list[Gate2EvaluationEntry] = []
    for number in (1, 2):
        audio = tmp_path / f"audio-{number}.wav"
        with wave.open(str(audio), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16_000)
            wav.writeframes(bytes([number, 0]) * 160)
        entries.append(
            Gate2EvaluationEntry(
                utterance_id=f"test-{number}",
                speaker_id=f"new-speaker-{number}",
                source_recording_id=f"recording-{number}",
                region="Busan",
                audio_filepath=audio.name,
                audio_sha256=_sha(audio),
                duration_seconds=0.01,
                surface_text=f"새 부산 발화 {number}",
                dialect_expressions=("발화",),
                label_status="approved",
            )
        )
    manifest = Gate2EvaluationManifest(
        dataset_id="independent-busan-test-v0",
        dataset_version="0.1.0",
        dataset_kind="independent_busan_test",
        source_name="test fixture",
        source_version="1",
        license_or_access_policy="test only",
        entries=tuple(entries),
    )
    manifest_path = tmp_path / "manifest.json"
    exclusions_path = tmp_path / "exclusions.json"
    _write_json(manifest_path, manifest)
    exclusions = EvaluationExclusionRegistry(
        source_artifacts=("fixture",),
        speaker_ids=(),
        utterance_ids=(),
        source_recording_ids=(),
        audio_sha256s=(),
        normalized_surface_texts=(),
    )
    _write_json(exclusions_path, exclusions)
    result = validate_evaluation_dataset(
        manifest_path=manifest_path,
        exclusion_path=exclusions_path,
        criteria=_criteria(),
    )
    assert result["passed"] is True

    _write_json(
        exclusions_path,
        exclusions.model_copy(update={"speaker_ids": ("new-speaker-1",)}),
    )
    result = validate_evaluation_dataset(
        manifest_path=manifest_path,
        exclusion_path=exclusions_path,
        criteria=_criteria(),
    )
    assert result["passed"] is False
    assert "speaker overlap: new-speaker-1" in result["errors"]


def test_prediction_comparison_uses_surface_text_for_rnnt_results(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\0\0" * 160)
    entry = Gate2EvaluationEntry(
        utterance_id="test-1",
        speaker_id="speaker-1",
        source_recording_id="recording-1",
        region="Busan",
        audio_filepath=audio.name,
        audio_sha256=_sha(audio),
        duration_seconds=0.01,
        surface_text="밥 묵었나",
        dialect_expressions=("묵었나",),
        label_status="approved",
    )
    manifest = Gate2EvaluationManifest(
        dataset_id="independent-busan-test-v0",
        dataset_version="1.0.0",
        dataset_kind="independent_busan_test",
        source_name="fixture",
        source_version="1",
        license_or_access_policy="test",
        entries=(entry,),
    )
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)

    def prediction(text: str, *, fine_tuned: bool, experiment: str) -> dict[str, Any]:
        return {
            "experiment_id": experiment,
            "benchmark_id": manifest.dataset_id,
            "benchmark_version": manifest.dataset_version,
            "utterance_id": entry.utterance_id,
            "audio_sha256": entry.audio_sha256,
            "result": {
                "surface_text": text,
                "confidence": None,
                "confidence_supported": False,
                "latency_ms": 10.0,
                "model": {
                    "name": "nemotron",
                    "version": "revision",
                    "model_family": "FastConformer-RNNT",
                    "decoder_type": "RNNT",
                    "target_language": "ko-KR",
                    "fine_tuned": fine_tuned,
                },
                "segments": [],
            },
        }

    pretrained_path = tmp_path / "pretrained.jsonl"
    fine_tuned_path = tmp_path / "fine-tuned.jsonl"
    pretrained_path.write_text(
        json.dumps(prediction("밤 무거나", fine_tuned=False, experiment="before")) + "\n",
        encoding="utf-8",
    )
    fine_tuned_path.write_text(
        json.dumps(prediction("밥 묵었나", fine_tuned=True, experiment="after")) + "\n",
        encoding="utf-8",
    )

    report = compare_gate2_predictions(
        manifest_path=manifest_path,
        pretrained_path=pretrained_path,
        fine_tuned_path=fine_tuned_path,
    )
    assert report["fine_tuned"]["cer"] == 0
    assert report["fine_tuned"]["dialect_preservation_rate"] == 1
    assert report["comparison"] == {"improved": 1, "equal": 0, "worsened": 0}


def test_gate2_current_evidence_passes_all_checks() -> None:
    evidence = Gate2Evidence.model_validate_json(
        (ROOT / "artifacts/gate-2/status/gate2-evidence.current.json").read_text(
            encoding="utf-8"
        )
    )
    assessment = assess_gate2(_criteria(), evidence)
    assert assessment.status == "PASS"
    assert assessment.pending_checks == ()
    assert assessment.failed_checks == ()
    assert "human_ab" in assessment.passed_checks
    assert "independent_cer_improvement" in assessment.passed_checks
    assert "independent_dialect_preservation" in assessment.passed_checks
    assert "standard_cer_regression" in assessment.passed_checks
    assert "reproducibility" in assessment.passed_checks
