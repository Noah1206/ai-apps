"""Safe, resumable import of TASK-004 single-speaker recordings."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

from busan_lab.audio import AudioProcessor, hash_file
from busan_lab.schemas.benchmark import BenchmarkManifest
from busan_lab.schemas.common import ConsentRecord, DatasetSplit, ReviewStatus
from busan_lab.schemas.training_import import (
    TrainingRecordingImportEntry,
    TrainingRecordingImportManifest,
    TrainingRecordingImportPlan,
    TrainingRecordingImportSummary,
    TrainingRecordingPlanItem,
    TrainingRecordingReviewDecision,
    TrainingRecordingReviewItem,
    TrainingRecordingReviewQueue,
)
from busan_lab.schemas.utterance import (
    LinguisticGroundTruth,
    SpeakerContext,
    UtteranceRecord,
)
from busan_lab.storage import LabStorage, RecordNotFoundError, file_is_dataless
from busan_lab.training import review_training_label

PROMPT_PATTERN = re.compile(r"^- (T004-S(\d{3})): (.+)$", re.MULTILINE)
RECORDING_PATTERN = re.compile(r"^New Recording(?: (\d+))?\.m4a$", re.IGNORECASE)


class TrainingRecordingImportError(ValueError):
    """Raised when a recording batch cannot be imported safely."""


def build_training_recording_import_plan(
    *,
    import_id: str,
    input_directory: Path,
    prompt_sheet: Path,
    prompt_start: int,
    prompt_end: int,
    speaker_id: str,
    region: str,
    device: str,
    recording_environment: str,
    consent: ConsentRecord,
    benchmark_manifests: Iterable[BenchmarkManifest],
) -> TrainingRecordingImportPlan:
    """Map numbered M4A recordings to prompt candidates without changing data."""

    source_directory = input_directory.expanduser().resolve()
    sheet_path = prompt_sheet.expanduser().resolve()
    if not source_directory.is_dir():
        raise FileNotFoundError(f"recording directory does not exist: {source_directory}")
    if not sheet_path.is_file():
        raise FileNotFoundError(f"prompt sheet does not exist: {sheet_path}")
    if prompt_start < 1 or prompt_end < prompt_start:
        raise ValueError("prompt range must satisfy 1 <= start <= end")

    expected_numbers = tuple(range(prompt_start, prompt_end + 1))
    prompt_text_by_number = _load_prompt_text(sheet_path)
    files_by_number: dict[int, Path] = {}
    errors: list[str] = []
    warnings: list[str] = []
    for discovered_path in sorted(source_directory.iterdir()):
        if not discovered_path.is_file() or discovered_path.name.startswith("."):
            continue
        if discovered_path.suffix.lower() != ".m4a":
            errors.append(f"unsupported file in recording directory: {discovered_path.name}")
            continue
        match = RECORDING_PATTERN.fullmatch(discovered_path.name)
        if match is None:
            errors.append(f"unexpected recording filename: {discovered_path.name}")
            continue
        number = int(match.group(1) or "1")
        if number in files_by_number:
            errors.append(f"duplicate recording number: {number}")
            continue
        files_by_number[number] = discovered_path

    missing_files = sorted(set(expected_numbers) - files_by_number.keys())
    unexpected_files = sorted(files_by_number.keys() - set(expected_numbers))
    missing_prompts = sorted(set(expected_numbers) - prompt_text_by_number.keys())
    if missing_files:
        errors.append(f"missing recording numbers: {missing_files}")
    if unexpected_files:
        errors.append(f"recordings outside requested range: {unexpected_files}")
    if missing_prompts:
        errors.append(f"missing prompt numbers: {missing_prompts}")

    frozen_benchmarks = tuple(
        benchmark for benchmark in benchmark_manifests if benchmark.frozen
    )
    checked = tuple(
        sorted(
            f"{benchmark.benchmark_id}@{benchmark.benchmark_version}"
            for benchmark in frozen_benchmarks
        )
    )
    if "busan-surface-v0@1.0.0" not in checked:
        errors.append("required frozen benchmark busan-surface-v0@1.0.0 was not checked")

    benchmark_speakers = {
        entry.speaker_id for benchmark in frozen_benchmarks for entry in benchmark.entries
    }
    if speaker_id in benchmark_speakers:
        errors.append(f"speaker {speaker_id!r} is already in a frozen benchmark")

    benchmark_audio_hashes = {
        audio_hash
        for benchmark in frozen_benchmarks
        for entry in benchmark.entries
        for audio_hash in (
            entry.original_audio_sha256,
            entry.derived_audio_sha256,
            *entry.lineage_audio_sha256s,
        )
    }
    benchmark_surfaces = {
        _surface_key(entry.surface_text)
        for benchmark in frozen_benchmarks
        for entry in benchmark.entries
    }

    if not consent.storage_allowed:
        errors.append("explicit storage consent is required")
    if not consent.research_use_allowed:
        errors.append("explicit research-use consent is required")
    if not consent.model_training_allowed:
        errors.append("explicit model-training consent is required")

    cloud_only_paths = tuple(
        path
        for number in expected_numbers
        if (path := files_by_number.get(number)) is not None
        and file_is_dataless(path)
    )
    if cloud_only_paths:
        errors.append(
            f"{len(cloud_only_paths)} recordings are cloud-only; download the "
            "busan_Audio folder to this Mac before import"
        )
    selected_paths = tuple(
        path
        for number in expected_numbers
        if (path := files_by_number.get(number)) is not None
        and path.stat().st_size > 0
        and path not in cloud_only_paths
    )
    hashes_by_path, hash_errors = _hash_recordings(selected_paths)
    errors.extend(hash_errors)
    items: list[TrainingRecordingPlanItem] = []
    for number in expected_numbers:
        mapped_path = files_by_number.get(number)
        prompt_text = prompt_text_by_number.get(number)
        if mapped_path is None or prompt_text is None:
            continue
        if mapped_path.stat().st_size <= 0:
            errors.append(f"recording is empty: {mapped_path.name}")
            continue
        if mapped_path in cloud_only_paths:
            continue
        source_hash = hashes_by_path.get(mapped_path)
        if source_hash is None:
            continue
        if source_hash in benchmark_audio_hashes:
            errors.append(f"recording leaks frozen Benchmark audio: {mapped_path.name}")
        if _surface_key(prompt_text) in benchmark_surfaces:
            errors.append(f"prompt leaks frozen Benchmark Surface text: T004-S{number:03d}")
        items.append(
            TrainingRecordingPlanItem(
                prompt_id=f"T004-S{number:03d}",
                source_filename=mapped_path.name,
                source_audio_sha256=source_hash,
                candidate_surface_text=prompt_text,
            )
        )

    hashes = [item.source_audio_sha256 for item in items]
    if len(hashes) != len(set(hashes)):
        errors.append("recording batch contains duplicate audio bytes")
    if len(items) != len(expected_numbers):
        errors.append(
            f"mapped {len(items)} recordings but expected {len(expected_numbers)}"
        )
    warnings.append(
        "prompt text remains candidate until a person reviews the converted audio"
    )
    warnings.append(
        "all recordings use one speaker and must remain in the train split"
    )
    return TrainingRecordingImportPlan(
        import_id=import_id,
        source_directory_name=source_directory.name,
        prompt_sheet_sha256=hash_file(sheet_path),
        speaker_id=speaker_id,
        region=region,
        device=device,
        recording_environment=recording_environment,
        consent=consent,
        expected_recordings=len(expected_numbers),
        items=tuple(items),
        benchmark_manifests_checked=checked,
        passed=not errors,
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(warnings),
    )


def execute_training_recording_import(
    *,
    plan: TrainingRecordingImportPlan,
    input_directory: Path,
    storage: LabStorage,
    processor: AudioProcessor,
) -> TrainingRecordingImportManifest:
    """Convert one validated batch and save candidate records plus an immutable ledger."""

    if not plan.passed:
        raise TrainingRecordingImportError("; ".join(plan.errors))
    if storage.training_recording_import_exists(plan.import_id):
        raise FileExistsError(f"training recording import already exists: {plan.import_id}")

    source_directory = input_directory.expanduser().resolve()
    records: list[UtteranceRecord] = []
    entries: list[TrainingRecordingImportEntry] = []
    for item in plan.items:
        source_path = source_directory / item.source_filename
        if not source_path.is_file() or file_is_dataless(source_path):
            raise FileNotFoundError(f"recording is unavailable: {source_path}")
        try:
            current_hash = hash_file(source_path)
        except OSError as error:
            raise TrainingRecordingImportError(
                f"recording cannot be read: {item.source_filename}: {error}"
            ) from error
        if current_hash != item.source_audio_sha256:
            raise TrainingRecordingImportError(
                f"recording changed after validation: {item.source_filename}"
            )
        utterance_id = _utterance_id(plan.import_id, item)
        try:
            existing = storage.load_utterance(utterance_id)
        except RecordNotFoundError:
            audio = processor.process(source_path, item.source_filename)
            record = UtteranceRecord(
                utterance_id=utterance_id,
                source="import",
                speaker=SpeakerContext(
                    speaker_id=plan.speaker_id,
                    region=plan.region,
                    device=plan.device,
                    environment=plan.recording_environment,
                ),
                dataset_split=DatasetSplit.TRAIN,
                consent=plan.consent,
                audio=audio,
                ground_truth=LinguisticGroundTruth(
                    surface_text=item.candidate_surface_text,
                    label_status=ReviewStatus.CANDIDATE,
                    label_version="label_v0",
                ),
            )
            records.append(record)
        else:
            _validate_resumable_record(existing, plan, item)
            record = existing
        entries.append(
            TrainingRecordingImportEntry(
                **item.model_dump(),
                utterance_id=record.utterance_id,
                duration_ms=record.audio.derived.duration_ms,
                audio_quality_passed=record.audio.quality.passed,
                audio_quality_warnings=record.audio.quality.warnings,
            )
        )

    failed_entries = [entry for entry in entries if not entry.audio_quality_passed]
    if failed_entries:
        failures = ", ".join(
            f"{entry.prompt_id}:{list(entry.audio_quality_warnings)}"
            for entry in failed_entries
        )
        raise TrainingRecordingImportError(
            f"audio quality failed; no new records were saved: {failures}"
        )
    for record in records:
        storage.save_utterance(record)
    manifest = TrainingRecordingImportManifest(
        import_id=plan.import_id,
        source_directory_name=plan.source_directory_name,
        prompt_sheet_sha256=plan.prompt_sheet_sha256,
        speaker_id=plan.speaker_id,
        region=plan.region,
        device=plan.device,
        recording_environment=plan.recording_environment,
        consent=plan.consent,
        entries=tuple(entries),
    )
    storage.save_training_recording_import(manifest)
    return manifest


def list_training_recording_import_summaries(
    storage: LabStorage,
) -> tuple[TrainingRecordingImportSummary, ...]:
    """List immutable recording imports without returning every queue entry."""

    return tuple(
        TrainingRecordingImportSummary(
            import_id=manifest.import_id,
            created_at=manifest.created_at,
            speaker_id=manifest.speaker_id,
            entry_count=len(manifest.entries),
        )
        for manifest in sorted(
            storage.list_training_recording_imports(),
            key=lambda item: item.created_at,
            reverse=True,
        )
    )


def build_training_recording_review_queue(
    storage: LabStorage,
    *,
    import_id: str,
) -> TrainingRecordingReviewQueue:
    """Join an immutable import ledger to each utterance's latest label state."""

    manifest = storage.load_training_recording_import(import_id)
    items = tuple(
        TrainingRecordingReviewItem(
            position=position,
            prompt_id=entry.prompt_id,
            utterance_id=entry.utterance_id,
            source_filename=entry.source_filename,
            candidate_surface_text=entry.candidate_surface_text,
            surface_text=record.ground_truth.surface_text,
            duration_ms=entry.duration_ms,
            audio_quality_passed=entry.audio_quality_passed,
            audio_quality_warnings=entry.audio_quality_warnings,
            label_status=record.ground_truth.label_status,
            label_version=record.ground_truth.label_version,
        )
        for position, entry in enumerate(manifest.entries, start=1)
        for record in (storage.load_utterance(entry.utterance_id),)
    )
    status_counts = {
        status: sum(item.label_status is status for item in items)
        for status in ReviewStatus
    }
    return TrainingRecordingReviewQueue(
        import_id=manifest.import_id,
        speaker_id=manifest.speaker_id,
        total_count=len(items),
        reviewed_count=len(items) - status_counts[ReviewStatus.CANDIDATE],
        candidate_count=status_counts[ReviewStatus.CANDIDATE],
        human_reviewed_count=status_counts[ReviewStatus.HUMAN_REVIEWED],
        approved_count=status_counts[ReviewStatus.APPROVED],
        rerecord_count=status_counts[ReviewStatus.DEPRECATED],
        items=items,
    )


def review_training_recording(
    storage: LabStorage,
    *,
    import_id: str,
    prompt_id: str,
    reviewer_id: str,
    decision: TrainingRecordingReviewDecision,
    notes: str | None,
) -> UtteranceRecord:
    """Append one review decision for an entry in a recording import."""

    manifest = storage.load_training_recording_import(import_id)
    entry = next(
        (item for item in manifest.entries if item.prompt_id == prompt_id),
        None,
    )
    if entry is None:
        raise RecordNotFoundError(f"{import_id}:{prompt_id}")
    status = (
        ReviewStatus.APPROVED
        if decision is TrainingRecordingReviewDecision.APPROVE
        else ReviewStatus.DEPRECATED
    )
    return review_training_label(
        storage,
        utterance_id=entry.utterance_id,
        reviewer_id=reviewer_id,
        status=status,
        reason=notes,
    )


def _load_prompt_text(prompt_sheet: Path) -> dict[int, str]:
    source = prompt_sheet.read_text(encoding="utf-8")
    prompts: dict[int, str] = {}
    for prompt_id, number_text, surface_text in PROMPT_PATTERN.findall(source):
        number = int(number_text)
        if number in prompts:
            raise ValueError(f"duplicate prompt number in sheet: {number}")
        if prompt_id != f"T004-S{number:03d}":
            raise ValueError(f"prompt ID is not canonical: {prompt_id}")
        prompts[number] = surface_text.strip()
    return prompts


def _hash_recordings(paths: tuple[Path, ...]) -> tuple[dict[Path, str], tuple[str, ...]]:
    if not paths:
        return {}, ()
    hashes: dict[Path, str] = {}
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=min(4, len(paths))) as executor:
        futures = {path: executor.submit(hash_file, path) for path in paths}
        for path in paths:
            try:
                hashes[path] = futures[path].result()
            except OSError as error:
                errors.append(f"recording cannot be read: {path.name}: {error}")
    return hashes, tuple(errors)


def _utterance_id(import_id: str, item: TrainingRecordingPlanItem) -> UUID:
    identity = f"busan-lab:{import_id}:{item.prompt_id}:{item.source_audio_sha256}"
    return uuid5(NAMESPACE_URL, identity)


def _validate_resumable_record(
    record: UtteranceRecord,
    plan: TrainingRecordingImportPlan,
    item: TrainingRecordingPlanItem,
) -> None:
    if (
        record.source != "import"
        or record.speaker.speaker_id != plan.speaker_id
        or record.dataset_split is not DatasetSplit.TRAIN
        or record.audio.original.sha256 != item.source_audio_sha256
        or record.ground_truth.surface_text != item.candidate_surface_text
        or record.ground_truth.label_status is not ReviewStatus.CANDIDATE
    ):
        raise TrainingRecordingImportError(
            f"existing deterministic record does not match import: {item.prompt_id}"
        )


def _surface_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())
