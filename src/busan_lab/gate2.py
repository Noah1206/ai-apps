"""Small command-line checks needed to close Gate 2 without model inference."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import wave
import zipfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from busan_lab.adapters.precomputed import PrecomputedSurfaceASRAdapter
from busan_lab.evaluation.metrics import character_error_rate, normalize_for_cer
from busan_lab.evaluation.runner import BenchmarkRunner
from busan_lab.schemas.asr import PrecomputedPrediction
from busan_lab.schemas.benchmark import BenchmarkManifest
from busan_lab.schemas.gate2 import (
    BenchmarkIntegrityAudit,
    BenchmarkSerialization,
    BlindABReviewItem,
    BlindABReviewResult,
    EvaluationExclusionRegistry,
    Gate2Assessment,
    Gate2Criteria,
    Gate2EvaluationManifest,
    Gate2Evidence,
    ReproducibilitySpec,
)
from busan_lab.storage import LabStorage


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_benchmark_bytes(manifest: BenchmarkManifest) -> bytes:
    """Serialize the validated semantic model independently of source formatting."""

    return json.dumps(
        manifest.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_benchmark_sha256(manifest: BenchmarkManifest) -> str:
    return hashlib.sha256(canonical_benchmark_bytes(manifest)).hexdigest()


def inspect_benchmark_serialization(
    path: Path,
    *,
    used_by: Sequence[str] = (),
) -> tuple[BenchmarkManifest, BenchmarkSerialization]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    manifest = BenchmarkManifest.model_validate(raw)
    raw_expression_count = sum(
        len(entry.get("dialect_expressions", ())) for entry in raw.get("entries", ())
    )
    semantic_expression_count = sum(len(entry.dialect_expressions) for entry in manifest.entries)
    structure: Literal["legacy_quoted_compound", "expanded_labels"] = (
        "legacy_quoted_compound"
        if raw_expression_count != semantic_expression_count
        else "expanded_labels"
    )
    return manifest, BenchmarkSerialization(
        path=str(path),
        raw_sha256=hash_file(path),
        semantic_sha256=canonical_benchmark_sha256(manifest),
        utterance_count=len(manifest.entries),
        raw_dialect_expression_count=raw_expression_count,
        semantic_dialect_expression_count=semantic_expression_count,
        annotation_structure=structure,
        used_by=tuple(used_by),
    )


def audit_benchmark_identity(
    *,
    storage_manifest_path: Path,
    canonical_manifest_path: Path,
    canonical_package_path: Path,
) -> BenchmarkIntegrityAudit:
    stored, stored_serialization = inspect_benchmark_serialization(
        storage_manifest_path,
        used_by=(
            "Audio Lab TASK-002/TASK-003A/TASK-003B evaluation after schema normalization",
        ),
    )
    stored_serialization = stored_serialization.model_copy(
        update={
            "related_artifacts": (
                "data/lab/experiments/task-002-nvidia-korean-conformer-ctc-pretrained-v0.json",
                "data/lab/experiments/task-003b-nemotron-3.5-asr-streaming-0.6b-pretrained-v0.json",
                "data/lab/reports/",
            )
        }
    )
    canonical, canonical_serialization = inspect_benchmark_serialization(
        canonical_manifest_path,
        used_by=(
            "TASK-003B external pretrained inference",
            "TASK-003C diagnostics",
            "TASK-005 pretrained-vs-adapter inference and automatic evaluation",
        ),
    )
    canonical_serialization = canonical_serialization.model_copy(
        update={
            "related_artifacts": (
                "artifacts/task-005/evaluation-v0/input/task-005-benchmark-v0/"
                "pretrained/predictions.jsonl",
                "artifacts/task-005/evaluation-v0/input/task-005-benchmark-v0/"
                "fine-tuned-best/predictions.jsonl",
                "artifacts/task-005/evaluation-v0/output/task-005-benchmark-comparison.json",
            )
        }
    )
    if stored != canonical:
        raise ValueError("benchmark serializations are not semantically equivalent")
    return BenchmarkIntegrityAudit(
        audit_id="gate-2-busan-surface-v0-1.0.0-semantic-identity-v1",
        benchmark_id=canonical.benchmark_id,
        benchmark_version=canonical.benchmark_version,
        decision="audited_semantic_equivalence_keep_version",
        canonical_artifact_path=str(canonical_package_path),
        canonical_package_sha256=hash_file(canonical_package_path),
        canonical_manifest_raw_sha256=canonical_serialization.raw_sha256,
        canonical_semantic_sha256=canonical_serialization.semantic_sha256,
        semantically_equivalent=True,
        serializations=(stored_serialization, canonical_serialization),
        provenance_note=(
            "The stored JSON preserves the user's quoted multi-expression input. "
            "BenchmarkEntry expands it during validation, so the exported JSON contains "
            "15 individual labels. Raw byte hashes differ; validated semantic content is equal. "
            "Both historical files remain unchanged and v1.0.0 remains canonical."
        ),
    )


def reevaluate_predictions(
    *,
    manifest_path: Path,
    prediction_paths: Sequence[Path],
) -> dict[str, Any]:
    manifest = BenchmarkManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    reports: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="gate2-reevaluation-") as directory:
        storage = LabStorage(Path(directory) / "lab")
        for prediction_path in prediction_paths:
            report = BenchmarkRunner(storage).run(
                manifest,
                PrecomputedSurfaceASRAdapter(prediction_path, manifest),
            )
            reports.append(
                {
                    "prediction_path": str(prediction_path),
                    "prediction_sha256": hash_file(prediction_path),
                    "experiment_id": report.experiment_id,
                    "metrics": report.metrics.model_dump(mode="json"),
                    "report": report.model_dump(mode="json"),
                }
            )
    return {
        "schema_version": "1.0.0",
        "evaluation_revision": "gate-2-canonical-semantic-reevaluation-v1",
        "inference_rerun": False,
        "benchmark_id": manifest.benchmark_id,
        "benchmark_version": manifest.benchmark_version,
        "benchmark_manifest_raw_sha256": hash_file(manifest_path),
        "benchmark_semantic_sha256": canonical_benchmark_sha256(manifest),
        "reports": reports,
    }


def verify_reproducibility_artifacts(root: Path, spec: ReproducibilitySpec) -> dict[str, Any]:
    resolved_root = root.resolve()
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for artifact in spec.artifacts:
        candidate = (resolved_root / artifact.relative_path).resolve()
        if candidate != resolved_root and resolved_root not in candidate.parents:
            errors.append(f"{artifact.artifact_id}: path escapes verification root")
            continue
        result: dict[str, Any] = {
            "artifact_id": artifact.artifact_id,
            "relative_path": artifact.relative_path,
            "required": artifact.required,
            "expected_sha256": artifact.expected_sha256,
            "exists": False,
            "actual_sha256": None,
            "verified": False,
        }
        if not candidate.exists():
            if artifact.required:
                errors.append(f"{artifact.artifact_id}: missing")
            results.append(result)
            continue
        if candidate.is_symlink() or not candidate.is_file():
            errors.append(f"{artifact.artifact_id}: must be a regular non-symlink file")
            results.append(result)
            continue
        result["exists"] = True
        actual_sha256 = hash_file(candidate)
        result["actual_sha256"] = actual_sha256
        if artifact.expected_sha256 is None:
            errors.append(f"{artifact.artifact_id}: recovered file hash is not pinned in spec")
        elif actual_sha256 != artifact.expected_sha256:
            errors.append(f"{artifact.artifact_id}: SHA-256 mismatch")
        else:
            result["verified"] = True
        results.append(result)
    return {
        "schema_version": "1.0.0",
        "task_id": spec.task_id,
        "experiment_id": spec.experiment_id,
        "passed": not errors,
        "errors": errors,
        "artifacts": results,
    }


def verify_recovery_bundle(
    bundle_root: Path,
    *,
    source_archive: Path | None = None,
) -> dict[str, Any]:
    """Verify every imported file against the bundle's own content index."""

    root = bundle_root.resolve()
    manifest_path = root / "MANIFEST.json"
    manifest_sha_path = root / "MANIFEST.sha256"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest_sha = manifest_sha_path.read_text(encoding="utf-8").split()[0]
    actual_manifest_sha = hash_file(manifest_path)
    errors: list[str] = []
    if expected_manifest_sha != actual_manifest_sha:
        errors.append("MANIFEST.json SHA-256 does not match MANIFEST.sha256")

    indexed: dict[str, dict[str, Any]] = {}
    for item in manifest.get("files", ()):
        relative = str(item.get("relative_path", ""))
        path = PurePosixPath(relative)
        if not relative or path.is_absolute() or ".." in path.parts or "\\" in relative:
            errors.append(f"unsafe manifest path: {relative}")
            continue
        if relative in indexed:
            errors.append(f"duplicate manifest path: {relative}")
            continue
        indexed[relative] = item

    verified = 0
    missing: list[str] = []
    mismatched: list[str] = []
    for relative, item in indexed.items():
        candidate = (root / relative).resolve()
        if root not in candidate.parents or candidate.is_symlink() or not candidate.is_file():
            missing.append(relative)
            continue
        if candidate.stat().st_size != int(item["size_bytes"]):
            mismatched.append(relative)
            continue
        if hash_file(candidate) != item["sha256"]:
            mismatched.append(relative)
            continue
        verified += 1
    errors.extend(f"missing or unsafe: {relative}" for relative in missing)
    errors.extend(f"size or SHA-256 mismatch: {relative}" for relative in mismatched)

    actual_files = {
        str(path.relative_to(root)).replace(os.sep, "/")
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    allowed_files = {*indexed, "MANIFEST.json", "MANIFEST.sha256"}
    unexpected = sorted(actual_files - allowed_files)
    if unexpected:
        errors.extend(f"unexpected imported file: {relative}" for relative in unexpected)

    archive_report: dict[str, Any] | None = None
    if source_archive is not None:
        archive_errors: list[str] = []
        member_count = 0
        with tarfile.open(source_archive, "r:gz") as archive:
            for member in archive:
                member_count += 1
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                    archive_errors.append(f"unsafe archive path: {member.name}")
                if not (member.isdir() or member.isfile()):
                    archive_errors.append(f"unsupported archive member: {member.name}")
        errors.extend(archive_errors)
        archive_report = {
            "sha256": hash_file(source_archive),
            "member_count": member_count,
            "safety_errors": archive_errors,
        }

    return {
        "schema_version": "1.0.0",
        "bundle_root": str(bundle_root),
        "manifest_sha256": actual_manifest_sha,
        "indexed_files": len(indexed),
        "verified_files": verified,
        "missing_files": missing,
        "mismatched_files": mismatched,
        "unexpected_files": unexpected,
        "source_archive": archive_report,
        "passed": not errors and verified == len(indexed),
        "errors": errors,
    }


def prepare_blind_ab_review(
    *,
    source_queue_path: Path,
    benchmark_path: Path,
    output_queue_path: Path,
    output_results_path: Path,
) -> tuple[dict[str, Any], BlindABReviewResult]:
    source = json.loads(source_queue_path.read_text(encoding="utf-8"))
    manifest = BenchmarkManifest.model_validate_json(benchmark_path.read_text(encoding="utf-8"))
    entries = {str(entry.utterance_id): entry for entry in manifest.entries}
    prepared_items: list[dict[str, Any]] = []
    result_items: list[BlindABReviewItem] = []
    for item in source["items"]:
        utterance_id = str(item["utterance_id"])
        entry = entries.get(utterance_id)
        if entry is None:
            raise ValueError(f"A/B item is not in canonical benchmark: {utterance_id}")
        source_audio = (source_queue_path.parent / item["audio_path"]).resolve()
        if not source_audio.is_file():
            raise FileNotFoundError(f"A/B audio is missing: {source_audio}")
        relative_audio = os.path.relpath(source_audio, output_queue_path.parent).replace(
            os.sep, "/"
        )
        prepared_items.append(
            {
                "item_id": item["item_id"],
                "utterance_id": utterance_id,
                "audio_path": relative_audio,
                "reference_surface_text": entry.surface_text,
                "dialect_expressions": [
                    label.surface_form for label in entry.dialect_expressions
                ],
                "candidate_A": item["A"],
                "candidate_B": item["B"],
            }
        )
        result_items.append(
            BlindABReviewItem(item_id=item["item_id"], utterance_id=utterance_id)
        )
    if len(prepared_items) != len(manifest.entries):
        raise ValueError("A/B queue must cover every benchmark utterance exactly once")
    queue = {
        "schema_version": "2.0.0",
        "status": "ready_for_blinded_review",
        "benchmark_id": manifest.benchmark_id,
        "benchmark_version": manifest.benchmark_version,
        "benchmark_semantic_sha256": canonical_benchmark_sha256(manifest),
        "model_names_exposed": False,
        "allowed_preferences": ["A", "B", "tie", "uncertain"],
        "allowed_findings": ["present", "absent", "uncertain"],
        "instructions": (
            "Listen to the WAV before judging. Do not open the key. Compare A and B for "
            "transcript closeness, dialect preservation, meaning fidelity, meaning distortion, "
            "and standard-Korean overcorrection."
        ),
        "items": prepared_items,
    }
    results = BlindABReviewResult(
        review_id="task-005-blinded-ab-review-v1",
        benchmark_id=manifest.benchmark_id,
        benchmark_version=manifest.benchmark_version,
        benchmark_semantic_sha256=canonical_benchmark_sha256(manifest),
        reviewer_id="REPLACE_WITH_REVIEWER_ID",
        status="in_progress",
        items=tuple(result_items),
    )
    return queue, results


def validate_blind_ab_review(
    *,
    queue_path: Path,
    results_path: Path,
    key_path: Path | None = None,
) -> dict[str, Any]:
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    serialized_queue = json.dumps(queue, ensure_ascii=False).casefold()
    forbidden_identifiers = ("nemotron", "pretrained", "fine_tuned", "nvidia")
    if any(identifier in serialized_queue for identifier in forbidden_identifiers):
        raise ValueError("blinded queue contains a model identifier")
    results = BlindABReviewResult.model_validate_json(results_path.read_text(encoding="utf-8"))
    queue_pairs = [(item["item_id"], str(item["utterance_id"])) for item in queue["items"]]
    result_pairs = [(item.item_id, item.utterance_id) for item in results.items]
    if queue_pairs != result_pairs:
        raise ValueError("review results do not match the blinded queue")
    decision_fields = (
        "transcript_preference",
        "dialect_preservation_preference",
        "meaning_fidelity_preference",
        "meaning_distortion_a",
        "meaning_distortion_b",
        "overcorrection_a",
        "overcorrection_b",
    )
    reviewed_items = sum(
        all(getattr(item, field) is not None for field in decision_fields)
        for item in results.items
    )
    complete = results.status == "complete"
    summary: dict[str, Any] | None = None
    if key_path is not None:
        if not complete:
            raise ValueError("the key cannot be used before all review dimensions are complete")
        key = json.loads(key_path.read_text(encoding="utf-8"))
        mappings = {item["item_id"]: item for item in key["mapping"]}
        wins = {"fine_tuned_best": 0, "pretrained": 0, "tie": 0, "uncertain": 0}
        for item in results.items:
            preference = item.transcript_preference
            if preference in {"tie", "uncertain"}:
                wins[preference] += 1
            else:
                wins[mappings[item.item_id][preference]] += 1
        decisive = wins["fine_tuned_best"] + wins["pretrained"]
        summary = {
            "transcript_preference": wins,
            "decisive_count": decisive,
            "fine_tuned_preference_rate": (
                wins["fine_tuned_best"] / decisive if decisive else None
            ),
        }
    return {
        "schema_version": "1.0.0",
        "valid": True,
        "complete": complete,
        "total_items": len(results.items),
        "reviewed_items": reviewed_items,
        "summary": summary,
    }


def _prompt_choice(
    prompt: str,
    choices: dict[str, str],
    *,
    input_fn: Any = input,
) -> str:
    while True:
        answer = input_fn(f"{prompt} ({'/'.join(choices)}): ").strip().casefold()
        if answer in choices:
            return choices[answer]
        print("허용된 선택지만 입력하세요.")


def _running_in_wsl() -> bool:
    if os.name != "posix" or sys.platform == "darwin":
        return False
    try:
        return "microsoft" in Path("/proc/sys/kernel/osrelease").read_text(
            encoding="utf-8"
        ).casefold()
    except OSError:
        return False


def _powershell_audio_command(audio_path: str) -> list[str]:
    escaped_path = audio_path.replace("'", "''")
    script = (
        "$ProgressPreference = 'SilentlyContinue'; "
        f"Start-Process -FilePath '{escaped_path}'"
    )
    encoded_script = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-EncodedCommand",
        encoded_script,
    ]


def _play_audio(audio_path: Path) -> None:
    if not audio_path.is_file():
        raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_path}")

    if sys.platform == "darwin":
        command = ["/usr/bin/open", str(audio_path)]
    elif os.name == "nt":
        command = _powershell_audio_command(str(audio_path))
    elif _running_in_wsl():
        windows_path = subprocess.run(
            ["wslpath", "-w", str(audio_path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        command = _powershell_audio_command(windows_path)
    elif xdg_open := shutil.which("xdg-open"):
        command = [xdg_open, str(audio_path)]
    elif ffplay := shutil.which("ffplay"):
        command = [
            ffplay,
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "error",
            str(audio_path),
        ]
    elif aplay := shutil.which("aplay"):
        command = [aplay, str(audio_path)]
    else:
        raise RuntimeError(
            "사용 가능한 오디오 재생기를 찾지 못했습니다. "
            "macOS open, Windows PowerShell, WSL 또는 xdg-open/ffplay/aplay가 필요합니다."
        )

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"오디오 파일 열기에 실패했습니다{suffix}") from error


def run_blind_ab_review(
    *,
    queue_path: Path,
    results_path: Path,
    reviewer_id: str,
    open_audio: bool = False,
    input_fn: Any = input,
) -> BlindABReviewResult:
    """Resume a model-blind terminal review and save after every item."""

    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    results = BlindABReviewResult.model_validate_json(
        results_path.read_text(encoding="utf-8")
    )
    queue_by_id = {item["item_id"]: item for item in queue["items"]}
    preference_choices = {"a": "A", "b": "B", "s": "tie", "u": "uncertain"}
    finding_choices = {
        "a": "A",
        "b": "B",
        "d": "both",
        "n": "neither",
        "u": "uncertain",
    }

    reviewed: list[BlindABReviewItem] = []
    for result_item in results.items:
        if result_item.transcript_preference is not None:
            reviewed.append(result_item)
            continue
        item = queue_by_id[result_item.item_id]
        audio_path = (queue_path.parent / item["audio_path"]).resolve()
        print(f"\n[{item['item_id']}] {item['reference_surface_text']}")
        print(f"방언 표현: {', '.join(item['dialect_expressions']) or '-'}")
        print(f"오디오: {audio_path}")
        print(f"A: {item['candidate_A']!r}")
        print(f"B: {item['candidate_B']!r}")
        if open_audio:
            _play_audio(audio_path)
            print("오디오를 기본 플레이어로 열었습니다. 재생을 확인한 뒤 판정하세요.")

        transcript = _prompt_choice(
            "실제 발화와 더 가까운 결과 [a=A, b=B, s=동일, u=판단불가]",
            preference_choices,
            input_fn=input_fn,
        )
        dialect = _prompt_choice(
            "부산 방언을 더 잘 보존한 결과 [a/b/s/u]",
            preference_choices,
            input_fn=input_fn,
        )
        meaning = _prompt_choice(
            "의미를 더 잘 보존한 결과 [a/b/s/u]",
            preference_choices,
            input_fn=input_fn,
        )
        distortion = _prompt_choice(
            "의미 왜곡이 있는 결과 [a=A, b=B, d=둘다, n=없음, u=불확실]",
            finding_choices,
            input_fn=input_fn,
        )
        overcorrection = _prompt_choice(
            "과도한 표준어 교정이 있는 결과 [a/b/d/n/u]",
            finding_choices,
            input_fn=input_fn,
        )

        def findings(value: str) -> tuple[str, str]:
            if value == "A":
                return "present", "absent"
            if value == "B":
                return "absent", "present"
            if value == "both":
                return "present", "present"
            if value == "neither":
                return "absent", "absent"
            return "uncertain", "uncertain"

        distortion_a, distortion_b = findings(distortion)
        overcorrection_a, overcorrection_b = findings(overcorrection)
        reviewed.append(
            result_item.model_copy(
                update={
                    "transcript_preference": transcript,
                    "dialect_preservation_preference": dialect,
                    "meaning_fidelity_preference": meaning,
                    "meaning_distortion_a": distortion_a,
                    "meaning_distortion_b": distortion_b,
                    "overcorrection_a": overcorrection_a,
                    "overcorrection_b": overcorrection_b,
                }
            )
        )
        progress = results.model_copy(
            update={
                "reviewer_id": reviewer_id,
                "items": tuple(reviewed) + results.items[len(reviewed) :],
            }
        )
        _write_json(results_path, progress)

    completed = results.model_copy(
        update={"reviewer_id": reviewer_id, "status": "complete", "items": tuple(reviewed)}
    )
    completed = BlindABReviewResult.model_validate(completed.model_dump(mode="json"))
    _write_json(results_path, completed)
    return completed


def _safe_zip_members(archive: zipfile.ZipFile) -> None:
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
            raise ValueError(f"unsafe ZIP member: {info.filename}")
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError(f"symlink ZIP member is not allowed: {info.filename}")


def _jsonl_from_zip(path: Path, member: str) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        _safe_zip_members(archive)
        lines = archive.read(member).decode("utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def build_exclusion_registry(
    *,
    train_package_path: Path,
    validation_package_path: Path,
    benchmark_path: Path,
) -> EvaluationExclusionRegistry:
    train_rows = _jsonl_from_zip(train_package_path, "train_manifest.jsonl")
    validation_rows = _jsonl_from_zip(validation_package_path, "validation_manifest.jsonl")
    benchmark = BenchmarkManifest.model_validate_json(benchmark_path.read_text(encoding="utf-8"))
    speakers = {row["speaker_id"] for row in (*train_rows, *validation_rows)}
    speakers.update(entry.speaker_id for entry in benchmark.entries)
    utterances = {str(row["utterance_id"]) for row in (*train_rows, *validation_rows)}
    utterances.update(str(entry.utterance_id) for entry in benchmark.entries)
    recording_ids = {str(row["utterance_id"]) for row in train_rows}
    recording_ids.update(str(row["utterance_id"]).split(".", 1)[0] for row in validation_rows)
    recording_ids.update(entry.original_audio_sha256 for entry in benchmark.entries)
    audio_hashes = {row["audio_sha256"] for row in (*train_rows, *validation_rows)}
    for entry in benchmark.entries:
        audio_hashes.update(
            (
                entry.original_audio_sha256,
                entry.derived_audio_sha256,
                *entry.lineage_audio_sha256s,
            )
        )
    surfaces = {
        normalize_for_cer(str(row["text"])) for row in (*train_rows, *validation_rows)
    }
    surfaces.update(normalize_for_cer(entry.surface_text) for entry in benchmark.entries)
    return EvaluationExclusionRegistry(
        source_artifacts=(
            f"{train_package_path}#sha256:{hash_file(train_package_path)}",
            f"{validation_package_path}#sha256:{hash_file(validation_package_path)}",
            f"{benchmark_path}#sha256:{hash_file(benchmark_path)}",
        ),
        speaker_ids=tuple(sorted(speakers)),
        utterance_ids=tuple(sorted(utterances)),
        source_recording_ids=tuple(sorted(recording_ids)),
        audio_sha256s=tuple(sorted(audio_hashes)),
        normalized_surface_texts=tuple(sorted(surfaces)),
    )


def validate_evaluation_dataset(
    *,
    manifest_path: Path,
    exclusion_path: Path,
    criteria: Gate2Criteria,
) -> dict[str, Any]:
    manifest = Gate2EvaluationManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    exclusions = EvaluationExclusionRegistry.model_validate_json(
        exclusion_path.read_text(encoding="utf-8")
    )
    excluded_speakers = set(exclusions.speaker_ids)
    excluded_utterances = set(exclusions.utterance_ids)
    excluded_recordings = set(exclusions.source_recording_ids)
    excluded_audio = set(exclusions.audio_sha256s)
    excluded_surfaces = set(exclusions.normalized_surface_texts)
    errors: list[str] = []
    warnings: list[str] = []
    speakers = {entry.speaker_id for entry in manifest.entries}
    minimum_utterances = (
        criteria.independent_min_utterances
        if manifest.dataset_kind == "independent_busan_test"
        else criteria.standard_min_utterances
    )
    minimum_speakers = (
        criteria.independent_min_speakers
        if manifest.dataset_kind == "independent_busan_test"
        else criteria.standard_min_speakers
    )
    if len(manifest.entries) < minimum_utterances:
        errors.append(f"requires at least {minimum_utterances} utterances")
    if len(speakers) < minimum_speakers:
        errors.append(f"requires at least {minimum_speakers} speakers")
    dataset_root = manifest_path.parent.resolve()
    for entry in manifest.entries:
        if entry.speaker_id in excluded_speakers:
            errors.append(f"speaker overlap: {entry.speaker_id}")
        if entry.utterance_id in excluded_utterances:
            errors.append(f"utterance overlap: {entry.utterance_id}")
        if entry.source_recording_id in excluded_recordings:
            errors.append(f"source recording overlap: {entry.source_recording_id}")
        lineage = {entry.audio_sha256, *entry.audio_lineage_sha256s}
        if lineage.intersection(excluded_audio):
            errors.append(f"audio lineage overlap: {entry.utterance_id}")
        if normalize_for_cer(entry.surface_text) in excluded_surfaces:
            warnings.append(f"exact transcript overlap: {entry.utterance_id}")
        audio_path = (dataset_root / entry.audio_filepath).resolve()
        if dataset_root not in audio_path.parents:
            errors.append(f"audio path escapes dataset root: {entry.audio_filepath}")
            continue
        if not audio_path.is_file():
            errors.append(f"missing audio: {entry.audio_filepath}")
            continue
        if hash_file(audio_path) != entry.audio_sha256:
            errors.append(f"audio SHA-256 mismatch: {entry.audio_filepath}")
            continue
        try:
            with wave.open(str(audio_path), "rb") as wav:
                if wav.getframerate() != 16_000 or wav.getnchannels() != 1:
                    errors.append(f"audio must be 16 kHz mono: {entry.audio_filepath}")
                if wav.getsampwidth() != 2 or wav.getnframes() <= 0:
                    errors.append(f"audio must be non-empty PCM16: {entry.audio_filepath}")
        except (wave.Error, EOFError):
            errors.append(f"corrupt WAV: {entry.audio_filepath}")
    if (
        manifest.dataset_kind == "independent_busan_test"
        and not any(entry.dialect_expressions for entry in manifest.entries)
    ):
        errors.append("independent Busan test requires reviewed dialect expressions")
    return {
        "schema_version": "1.0.0",
        "dataset_id": manifest.dataset_id,
        "dataset_version": manifest.dataset_version,
        "dataset_kind": manifest.dataset_kind,
        "passed": not errors,
        "utterances": len(manifest.entries),
        "speakers": len(speakers),
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
    }


def _load_prediction_pair(
    path: Path,
    manifest: Gate2EvaluationManifest,
    *,
    expected_fine_tuned: bool,
) -> tuple[str, dict[str, PrecomputedPrediction]]:
    records = [
        PrecomputedPrediction.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != len(manifest.entries):
        raise ValueError(f"prediction count does not match dataset: {path}")
    if any(record.result.model.fine_tuned is not expected_fine_tuned for record in records):
        raise ValueError(f"prediction fine_tuned metadata is wrong: {path}")
    if any(
        record.benchmark_id != manifest.dataset_id
        or record.benchmark_version != manifest.dataset_version
        for record in records
    ):
        raise ValueError(f"prediction dataset identity is wrong: {path}")
    by_id = {record.utterance_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError(f"duplicate prediction utterance ID: {path}")
    expected = {entry.utterance_id: entry.audio_sha256 for entry in manifest.entries}
    actual = {record.utterance_id: record.audio_sha256 for record in records}
    if actual != expected:
        raise ValueError(f"prediction IDs or audio hashes do not match dataset: {path}")
    experiment_ids = {record.experiment_id for record in records}
    if len(experiment_ids) != 1:
        raise ValueError(f"prediction file must contain one experiment: {path}")
    return experiment_ids.pop(), by_id


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def compare_gate2_predictions(
    *,
    manifest_path: Path,
    pretrained_path: Path,
    fine_tuned_path: Path,
) -> dict[str, Any]:
    manifest = Gate2EvaluationManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    pretrained_id, pretrained = _load_prediction_pair(
        pretrained_path, manifest, expected_fine_tuned=False
    )
    fine_tuned_id, fine_tuned = _load_prediction_pair(
        fine_tuned_path, manifest, expected_fine_tuned=True
    )
    rows: list[dict[str, Any]] = []
    totals = {
        "pretrained_edits": 0,
        "fine_tuned_edits": 0,
        "reference_characters": 0,
        "dialect_expressions": 0,
        "pretrained_preserved": 0,
        "fine_tuned_preserved": 0,
    }
    outcomes = {"improved": 0, "equal": 0, "worsened": 0}
    for entry in manifest.entries:
        before = pretrained[entry.utterance_id]
        after = fine_tuned[entry.utterance_id]
        before_cer, before_edits = character_error_rate(
            entry.surface_text, before.result.surface_text
        )
        after_cer, after_edits = character_error_rate(
            entry.surface_text, after.result.surface_text
        )
        outcome = (
            "improved"
            if after_cer < before_cer
            else "worsened"
            if after_cer > before_cer
            else "equal"
        )
        outcomes[outcome] += 1
        before_preserved = sum(
            expression in before.result.surface_text for expression in entry.dialect_expressions
        )
        after_preserved = sum(
            expression in after.result.surface_text for expression in entry.dialect_expressions
        )
        totals["pretrained_edits"] += (
            before_edits.substitutions + before_edits.deletions + before_edits.insertions
        )
        totals["fine_tuned_edits"] += (
            after_edits.substitutions + after_edits.deletions + after_edits.insertions
        )
        totals["reference_characters"] += before_edits.reference_characters
        totals["dialect_expressions"] += len(entry.dialect_expressions)
        totals["pretrained_preserved"] += before_preserved
        totals["fine_tuned_preserved"] += after_preserved
        rows.append(
            {
                "utterance_id": entry.utterance_id,
                "speaker_id": entry.speaker_id,
                "reference_surface_text": entry.surface_text,
                "pretrained_surface_text": before.result.surface_text,
                "fine_tuned_surface_text": after.result.surface_text,
                "pretrained_cer": before_cer,
                "fine_tuned_cer": after_cer,
                "outcome": outcome,
                "pretrained_empty": not before.result.surface_text.strip(),
                "fine_tuned_empty": not after.result.surface_text.strip(),
                "pretrained_latency_ms": before.result.latency_ms,
                "fine_tuned_latency_ms": after.result.latency_ms,
                "dialect_expression_count": len(entry.dialect_expressions),
                "pretrained_preserved_count": before_preserved,
                "fine_tuned_preserved_count": after_preserved,
            }
        )

    reference_characters = totals["reference_characters"]
    expression_count = totals["dialect_expressions"]

    def metrics(
        arm: Literal["pretrained", "fine_tuned"],
        experiment_id: str,
        prediction_path: Path,
    ) -> dict[str, Any]:
        latencies = [float(row[f"{arm}_latency_ms"]) for row in rows]
        return {
            "experiment_id": experiment_id,
            "predictions_sha256": hash_file(prediction_path),
            "cer": totals[f"{arm}_edits"] / reference_characters,
            "dialect_preservation_rate": (
                totals[f"{arm}_preserved"] / expression_count
                if manifest.dataset_kind == "independent_busan_test" and expression_count
                else None
            ),
            "empty_outputs": sum(bool(row[f"{arm}_empty"]) for row in rows),
            "latency_mean_ms": statistics.fmean(latencies),
            "latency_p50_ms": statistics.median(latencies),
            "latency_p95_ms": _percentile(latencies, 0.95),
        }

    return {
        "schema_version": "1.0.0",
        "dataset_id": manifest.dataset_id,
        "dataset_version": manifest.dataset_version,
        "dataset_kind": manifest.dataset_kind,
        "manifest_sha256": hash_file(manifest_path),
        "utterances": len(manifest.entries),
        "speakers": len({entry.speaker_id for entry in manifest.entries}),
        "normalization": "busan_lab.evaluation.metrics.normalize_for_cer",
        "pretrained": metrics("pretrained", pretrained_id, pretrained_path),
        "fine_tuned": metrics("fine_tuned", fine_tuned_id, fine_tuned_path),
        "comparison": outcomes,
        "utterance_comparisons": rows,
    }


def refresh_gate2_evidence(
    *,
    base: Gate2Evidence,
    reproducibility_report: dict[str, Any],
    bundle_report: dict[str, Any],
    queue_path: Path,
    results_path: Path,
    key_path: Path | None,
    independent_report: dict[str, Any] | None = None,
    standard_report: dict[str, Any] | None = None,
) -> Gate2Evidence:
    human_status = validate_blind_ab_review(
        queue_path=queue_path,
        results_path=results_path,
        key_path=key_path,
    )
    if human_status["complete"] and human_status["summary"] is None:
        raise ValueError("completed human review requires the blinded key")
    human_rate = (
        human_status["summary"]["fine_tuned_preference_rate"]
        if human_status["summary"] is not None
        else None
    )
    updates: dict[str, Any] = {
        "reproducibility_passed": bool(
            reproducibility_report.get("passed") and bundle_report.get("passed")
        ),
        "human_review_complete": bool(human_status["complete"]),
        "human_fine_tuned_preference_rate": human_rate,
    }

    if independent_report is not None:
        updates.update(
            {
                "independent_test_complete": True,
                "independent_utterances": independent_report["utterances"],
                "independent_speakers": independent_report["speakers"],
                "independent_pretrained_cer": independent_report["pretrained"]["cer"],
                "independent_fine_tuned_cer": independent_report["fine_tuned"]["cer"],
                "independent_pretrained_dialect_preservation": independent_report[
                    "pretrained"
                ]["dialect_preservation_rate"],
                "independent_fine_tuned_dialect_preservation": independent_report[
                    "fine_tuned"
                ]["dialect_preservation_rate"],
                "independent_pretrained_empty_outputs": independent_report["pretrained"][
                    "empty_outputs"
                ],
                "independent_fine_tuned_empty_outputs": independent_report["fine_tuned"][
                    "empty_outputs"
                ],
            }
        )
    if standard_report is not None:
        updates.update(
            {
                "standard_regression_complete": True,
                "standard_utterances": standard_report["utterances"],
                "standard_speakers": standard_report["speakers"],
                "standard_pretrained_cer": standard_report["pretrained"]["cer"],
                "standard_fine_tuned_cer": standard_report["fine_tuned"]["cer"],
                "standard_pretrained_empty_outputs": standard_report["pretrained"][
                    "empty_outputs"
                ],
                "standard_fine_tuned_empty_outputs": standard_report["fine_tuned"][
                    "empty_outputs"
                ],
            }
        )
    return Gate2Evidence.model_validate(base.model_copy(update=updates).model_dump(mode="json"))


def evaluate_gate2_suite(
    *,
    independent_manifest: Path,
    independent_pretrained: Path,
    independent_fine_tuned: Path,
    standard_manifest: Path,
    standard_pretrained: Path,
    standard_fine_tuned: Path,
    exclusions_path: Path,
    criteria: Gate2Criteria,
    base_evidence: Gate2Evidence,
    reproducibility_report: dict[str, Any],
    bundle_report: dict[str, Any],
    queue_path: Path,
    results_path: Path,
    key_path: Path,
    output_dir: Path,
) -> Gate2Assessment:
    """Validate both held-out sets, compare predictions, and reassess Gate 2."""

    validations = {
        "independent_busan": validate_evaluation_dataset(
            manifest_path=independent_manifest,
            exclusion_path=exclusions_path,
            criteria=criteria,
        ),
        "standard_korean": validate_evaluation_dataset(
            manifest_path=standard_manifest,
            exclusion_path=exclusions_path,
            criteria=criteria,
        ),
    }
    failed = [name for name, result in validations.items() if not result["passed"]]
    if failed:
        raise ValueError(f"evaluation dataset validation failed: {', '.join(failed)}")

    independent_report = compare_gate2_predictions(
        manifest_path=independent_manifest,
        pretrained_path=independent_pretrained,
        fine_tuned_path=independent_fine_tuned,
    )
    standard_report = compare_gate2_predictions(
        manifest_path=standard_manifest,
        pretrained_path=standard_pretrained,
        fine_tuned_path=standard_fine_tuned,
    )
    review = BlindABReviewResult.model_validate_json(
        results_path.read_text(encoding="utf-8")
    )
    evidence = refresh_gate2_evidence(
        base=base_evidence,
        reproducibility_report=reproducibility_report,
        bundle_report=bundle_report,
        queue_path=queue_path,
        results_path=results_path,
        key_path=key_path if review.status == "complete" else None,
        independent_report=independent_report,
        standard_report=standard_report,
    )
    assessment = assess_gate2(criteria, evidence)
    _write_json(output_dir / "dataset-validation.json", validations)
    _write_json(output_dir / "independent-busan-comparison.json", independent_report)
    _write_json(output_dir / "standard-korean-comparison.json", standard_report)
    _write_json(output_dir / "gate2-evidence.json", evidence)
    _write_json(output_dir / "gate2-assessment.json", assessment)
    return assessment


def assess_gate2(criteria: Gate2Criteria, evidence: Gate2Evidence) -> Gate2Assessment:
    passed: list[str] = []
    failed: list[str] = []
    pending: list[str] = []

    def require(name: str, condition: bool) -> None:
        (passed if condition else failed).append(name)

    require("benchmark_integrity", evidence.benchmark_integrity_passed)
    require("reproducibility", evidence.reproducibility_passed)
    require("pilot_cer", evidence.pilot_fine_tuned_cer <= criteria.pilot_max_cer)
    require(
        "pilot_dialect_preservation",
        evidence.pilot_fine_tuned_dialect_preservation
        >= criteria.pilot_min_dialect_preservation,
    )
    require(
        "pilot_empty_output_regression",
        evidence.pilot_fine_tuned_empty_outputs
        <= evidence.pilot_pretrained_empty_outputs + criteria.empty_output_increase_allowed,
    )
    if not evidence.human_review_complete:
        pending.append("human_ab")
    else:
        require(
            "human_ab",
            evidence.human_fine_tuned_preference_rate is not None
            and evidence.human_fine_tuned_preference_rate
            >= criteria.human_min_fine_tuned_preference_rate,
        )
    if not evidence.independent_test_complete:
        pending.append("independent_multi_speaker_test")
    else:
        independent_values = (
            evidence.independent_pretrained_cer,
            evidence.independent_fine_tuned_cer,
            evidence.independent_pretrained_dialect_preservation,
            evidence.independent_fine_tuned_dialect_preservation,
            evidence.independent_pretrained_empty_outputs,
            evidence.independent_fine_tuned_empty_outputs,
        )
        if any(value is None for value in independent_values):
            failed.append("independent_metrics_complete")
        else:
            assert evidence.independent_pretrained_cer is not None
            assert evidence.independent_fine_tuned_cer is not None
            assert evidence.independent_pretrained_dialect_preservation is not None
            assert evidence.independent_fine_tuned_dialect_preservation is not None
            assert evidence.independent_pretrained_empty_outputs is not None
            assert evidence.independent_fine_tuned_empty_outputs is not None
            pretrained_cer = evidence.independent_pretrained_cer
            fine_cer = evidence.independent_fine_tuned_cer
            relative_improvement = (
                (pretrained_cer - fine_cer) / pretrained_cer
                if pretrained_cer > 0
                else float(fine_cer == 0)
            )
            require(
                "independent_dataset_size",
                evidence.independent_utterances >= criteria.independent_min_utterances
                and evidence.independent_speakers >= criteria.independent_min_speakers,
            )
            require(
                "independent_cer_improvement",
                relative_improvement >= criteria.independent_min_relative_cer_improvement,
            )
            require(
                "independent_dialect_preservation",
                evidence.independent_fine_tuned_dialect_preservation
                - evidence.independent_pretrained_dialect_preservation
                >= criteria.independent_min_dialect_preservation_delta,
            )
            require(
                "independent_empty_output_regression",
                evidence.independent_fine_tuned_empty_outputs
                <= evidence.independent_pretrained_empty_outputs
                + criteria.empty_output_increase_allowed,
            )
    if not evidence.standard_regression_complete:
        pending.append("standard_korean_regression")
    else:
        if (
            evidence.standard_pretrained_cer is None
            or evidence.standard_fine_tuned_cer is None
            or evidence.standard_pretrained_empty_outputs is None
            or evidence.standard_fine_tuned_empty_outputs is None
        ):
            failed.append("standard_metrics_complete")
        else:
            allowed_regression = max(
                criteria.standard_max_absolute_cer_regression,
                evidence.standard_pretrained_cer
                * criteria.standard_max_relative_cer_regression,
            )
            require(
                "standard_dataset_size",
                evidence.standard_utterances >= criteria.standard_min_utterances
                and evidence.standard_speakers >= criteria.standard_min_speakers,
            )
            require(
                "standard_cer_regression",
                evidence.standard_fine_tuned_cer - evidence.standard_pretrained_cer
                <= allowed_regression,
            )
            require(
                "standard_empty_output_regression",
                evidence.standard_fine_tuned_empty_outputs
                <= evidence.standard_pretrained_empty_outputs
                + criteria.empty_output_increase_allowed,
            )
    critical = {"benchmark_integrity", "reproducibility"}
    safety = {
        "pilot_empty_output_regression",
        "independent_empty_output_regression",
        "standard_cer_regression",
        "standard_empty_output_regression",
    }
    if pending or critical.intersection(failed) or safety.intersection(failed):
        status: Literal["PASS", "CONDITIONAL PASS", "FAIL"] = "FAIL"
    elif not failed:
        status = "PASS"
    elif len(failed) == 1:
        status = "CONDITIONAL PASS"
    else:
        status = "FAIL"
    return Gate2Assessment(
        status=status,
        passed_checks=tuple(passed),
        failed_checks=tuple(failed),
        pending_checks=tuple(pending),
        note=(
            "FAIL means Gate 2 evidence is incomplete or a safety/integrity check failed; "
            "it does not by itself mean the adapter weights are unusable. Thresholds are proposals."
        ),
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("benchmark-audit")
    audit.add_argument("--storage-manifest", type=Path, required=True)
    audit.add_argument("--canonical-manifest", type=Path, required=True)
    audit.add_argument("--canonical-package", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)

    reevaluate = commands.add_parser("reevaluate")
    reevaluate.add_argument("--manifest", type=Path, required=True)
    reevaluate.add_argument("--predictions", type=Path, action="append", required=True)
    reevaluate.add_argument("--output", type=Path, required=True)

    verify = commands.add_parser("verify-artifacts")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--spec", type=Path, required=True)
    verify.add_argument("--output", type=Path, default=None)

    verify_bundle = commands.add_parser("verify-recovery-bundle")
    verify_bundle.add_argument("--bundle-root", type=Path, required=True)
    verify_bundle.add_argument("--source-archive", type=Path, default=None)
    verify_bundle.add_argument("--output", type=Path, default=None)

    prepare_ab = commands.add_parser("prepare-ab")
    prepare_ab.add_argument("--source-queue", type=Path, required=True)
    prepare_ab.add_argument("--benchmark", type=Path, required=True)
    prepare_ab.add_argument("--output-queue", type=Path, required=True)
    prepare_ab.add_argument("--output-results", type=Path, required=True)

    validate_ab = commands.add_parser("validate-ab")
    validate_ab.add_argument("--queue", type=Path, required=True)
    validate_ab.add_argument("--results", type=Path, required=True)
    validate_ab.add_argument("--key", type=Path, default=None)

    review_ab = commands.add_parser("review-ab")
    review_ab.add_argument("--queue", type=Path, required=True)
    review_ab.add_argument("--results", type=Path, required=True)
    review_ab.add_argument("--reviewer-id", required=True)
    review_ab.add_argument("--open-audio", action="store_true")

    exclusions = commands.add_parser("build-exclusions")
    exclusions.add_argument("--train-package", type=Path, required=True)
    exclusions.add_argument("--validation-package", type=Path, required=True)
    exclusions.add_argument("--benchmark", type=Path, required=True)
    exclusions.add_argument("--output", type=Path, required=True)

    dataset = commands.add_parser("validate-dataset")
    dataset.add_argument("--manifest", type=Path, required=True)
    dataset.add_argument("--exclusions", type=Path, required=True)
    dataset.add_argument("--criteria", type=Path, required=True)
    dataset.add_argument("--output", type=Path, default=None)

    compare = commands.add_parser("compare-predictions")
    compare.add_argument("--manifest", type=Path, required=True)
    compare.add_argument("--pretrained", type=Path, required=True)
    compare.add_argument("--fine-tuned", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)

    suite = commands.add_parser("evaluate-suite")
    suite.add_argument("--independent-manifest", type=Path, required=True)
    suite.add_argument("--independent-pretrained", type=Path, required=True)
    suite.add_argument("--independent-fine-tuned", type=Path, required=True)
    suite.add_argument("--standard-manifest", type=Path, required=True)
    suite.add_argument("--standard-pretrained", type=Path, required=True)
    suite.add_argument("--standard-fine-tuned", type=Path, required=True)
    suite.add_argument("--exclusions", type=Path, required=True)
    suite.add_argument("--criteria", type=Path, required=True)
    suite.add_argument("--base-evidence", type=Path, required=True)
    suite.add_argument("--repro-verification", type=Path, required=True)
    suite.add_argument("--bundle-verification", type=Path, required=True)
    suite.add_argument("--queue", type=Path, required=True)
    suite.add_argument("--results", type=Path, required=True)
    suite.add_argument("--key", type=Path, required=True)
    suite.add_argument("--output-dir", type=Path, required=True)

    refresh = commands.add_parser("refresh-assessment")
    refresh.add_argument("--base-evidence", type=Path, required=True)
    refresh.add_argument("--criteria", type=Path, required=True)
    refresh.add_argument("--repro-verification", type=Path, required=True)
    refresh.add_argument("--bundle-verification", type=Path, required=True)
    refresh.add_argument("--queue", type=Path, required=True)
    refresh.add_argument("--results", type=Path, required=True)
    refresh.add_argument("--key", type=Path, default=None)
    refresh.add_argument("--independent-report", type=Path, default=None)
    refresh.add_argument("--standard-report", type=Path, default=None)
    refresh.add_argument("--evidence-output", type=Path, required=True)
    refresh.add_argument("--assessment-output", type=Path, required=True)

    assess = commands.add_parser("assess")
    assess.add_argument("--criteria", type=Path, required=True)
    assess.add_argument("--evidence", type=Path, required=True)
    assess.add_argument("--output", type=Path, default=None)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(arguments)
    result: Any
    if parsed.command == "benchmark-audit":
        result = audit_benchmark_identity(
            storage_manifest_path=parsed.storage_manifest,
            canonical_manifest_path=parsed.canonical_manifest,
            canonical_package_path=parsed.canonical_package,
        )
        _write_json(parsed.output, result)
        print(result.model_dump_json(indent=2))
        return 0
    if parsed.command == "reevaluate":
        result = reevaluate_predictions(
            manifest_path=parsed.manifest,
            prediction_paths=parsed.predictions,
        )
        _write_json(parsed.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if parsed.command == "verify-artifacts":
        spec = ReproducibilitySpec.model_validate_json(parsed.spec.read_text(encoding="utf-8"))
        result = verify_reproducibility_artifacts(parsed.root, spec)
        if parsed.output is not None:
            _write_json(parsed.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["passed"] else 1
    if parsed.command == "verify-recovery-bundle":
        result = verify_recovery_bundle(
            parsed.bundle_root,
            source_archive=parsed.source_archive,
        )
        if parsed.output is not None:
            _write_json(parsed.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["passed"] else 1
    if parsed.command == "prepare-ab":
        queue, results = prepare_blind_ab_review(
            source_queue_path=parsed.source_queue,
            benchmark_path=parsed.benchmark,
            output_queue_path=parsed.output_queue,
            output_results_path=parsed.output_results,
        )
        _write_json(parsed.output_queue, queue)
        _write_json(parsed.output_results, results)
        print(parsed.output_queue)
        print(parsed.output_results)
        return 0
    if parsed.command == "validate-ab":
        result = validate_blind_ab_review(
            queue_path=parsed.queue,
            results_path=parsed.results,
            key_path=parsed.key,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if parsed.command == "review-ab":
        result = run_blind_ab_review(
            queue_path=parsed.queue,
            results_path=parsed.results,
            reviewer_id=parsed.reviewer_id,
            open_audio=parsed.open_audio,
        )
        print(result.model_dump_json(indent=2))
        return 0
    if parsed.command == "build-exclusions":
        result = build_exclusion_registry(
            train_package_path=parsed.train_package,
            validation_package_path=parsed.validation_package,
            benchmark_path=parsed.benchmark,
        )
        _write_json(parsed.output, result)
        print(result.model_dump_json(indent=2))
        return 0
    if parsed.command == "validate-dataset":
        criteria = Gate2Criteria.model_validate_json(
            parsed.criteria.read_text(encoding="utf-8")
        )
        result = validate_evaluation_dataset(
            manifest_path=parsed.manifest,
            exclusion_path=parsed.exclusions,
            criteria=criteria,
        )
        if parsed.output is not None:
            _write_json(parsed.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["passed"] else 1
    if parsed.command == "compare-predictions":
        result = compare_gate2_predictions(
            manifest_path=parsed.manifest,
            pretrained_path=parsed.pretrained,
            fine_tuned_path=parsed.fine_tuned,
        )
        _write_json(parsed.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if parsed.command == "evaluate-suite":
        criteria = Gate2Criteria.model_validate_json(
            parsed.criteria.read_text(encoding="utf-8")
        )
        evidence = Gate2Evidence.model_validate_json(
            parsed.base_evidence.read_text(encoding="utf-8")
        )
        result = evaluate_gate2_suite(
            independent_manifest=parsed.independent_manifest,
            independent_pretrained=parsed.independent_pretrained,
            independent_fine_tuned=parsed.independent_fine_tuned,
            standard_manifest=parsed.standard_manifest,
            standard_pretrained=parsed.standard_pretrained,
            standard_fine_tuned=parsed.standard_fine_tuned,
            exclusions_path=parsed.exclusions,
            criteria=criteria,
            base_evidence=evidence,
            reproducibility_report=json.loads(
                parsed.repro_verification.read_text(encoding="utf-8")
            ),
            bundle_report=json.loads(
                parsed.bundle_verification.read_text(encoding="utf-8")
            ),
            queue_path=parsed.queue,
            results_path=parsed.results,
            key_path=parsed.key,
            output_dir=parsed.output_dir,
        )
        print(result.model_dump_json(indent=2))
        return 0 if result.status == "PASS" else 1
    if parsed.command == "refresh-assessment":
        criteria = Gate2Criteria.model_validate_json(
            parsed.criteria.read_text(encoding="utf-8")
        )
        base = Gate2Evidence.model_validate_json(
            parsed.base_evidence.read_text(encoding="utf-8")
        )
        result_evidence = refresh_gate2_evidence(
            base=base,
            reproducibility_report=json.loads(
                parsed.repro_verification.read_text(encoding="utf-8")
            ),
            bundle_report=json.loads(
                parsed.bundle_verification.read_text(encoding="utf-8")
            ),
            queue_path=parsed.queue,
            results_path=parsed.results,
            key_path=parsed.key,
            independent_report=(
                json.loads(parsed.independent_report.read_text(encoding="utf-8"))
                if parsed.independent_report is not None
                else None
            ),
            standard_report=(
                json.loads(parsed.standard_report.read_text(encoding="utf-8"))
                if parsed.standard_report is not None
                else None
            ),
        )
        result_assessment = assess_gate2(criteria, result_evidence)
        _write_json(parsed.evidence_output, result_evidence)
        _write_json(parsed.assessment_output, result_assessment)
        print(result_assessment.model_dump_json(indent=2))
        return 0 if result_assessment.status == "PASS" else 1
    if parsed.command == "assess":
        criteria = Gate2Criteria.model_validate_json(
            parsed.criteria.read_text(encoding="utf-8")
        )
        evidence = Gate2Evidence.model_validate_json(parsed.evidence.read_text(encoding="utf-8"))
        result = assess_gate2(criteria, evidence)
        if parsed.output is not None:
            _write_json(parsed.output, result)
        print(result.model_dump_json(indent=2))
        return 0 if result.status == "PASS" else 1
    raise AssertionError(parsed.command)


if __name__ == "__main__":
    raise SystemExit(main())
