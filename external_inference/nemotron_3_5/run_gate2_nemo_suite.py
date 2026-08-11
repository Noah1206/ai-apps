#!/usr/bin/env python3
"""Run the recovery-pinned NeMo runner for both Gate 2 datasets and model arms."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

PINNED_RUNNER_SHA256 = "88232b4e2eae22608e5fb7144b9518f388c7ea5d4c2e75dda14bb273eb37ad7f"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_dataset(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("frozen") is not True
        or manifest.get("training_allowed") is not False
        or manifest.get("checkpoint_selection_allowed") is not False
    ):
        raise RuntimeError(f"held-out dataset gates failed: {path}")
    if manifest.get("dataset_kind") not in {
        "independent_busan_test",
        "standard_korean_regression",
    }:
        raise RuntimeError(f"unsupported Gate 2 dataset kind: {path}")
    root = path.parent.resolve()
    entries: list[dict[str, Any]] = []
    for item in manifest.get("entries", ()):
        relative = str(item["audio_filepath"])
        pure_path = PurePosixPath(relative)
        if pure_path.is_absolute() or ".." in pure_path.parts or "\\" in relative:
            raise RuntimeError(f"unsafe audio path: {relative}")
        audio_path = (root / relative).resolve()
        if root not in audio_path.parents or not audio_path.is_file():
            raise RuntimeError(f"missing or unsafe audio: {relative}")
        if sha256_file(audio_path) != item["audio_sha256"]:
            raise RuntimeError(f"audio SHA-256 mismatch: {item['utterance_id']}")
        entries.append(
            {
                "utterance_id": str(item["utterance_id"]),
                "audio_sha256": item["audio_sha256"],
                "audio_path": audio_path,
            }
        )
    if not entries or len({item["utterance_id"] for item in entries}) != len(entries):
        raise RuntimeError(f"empty dataset or duplicate utterance ID: {path}")
    return manifest, entries


def load_runner(path: Path) -> ModuleType:
    if sha256_file(path) != PINNED_RUNNER_SHA256:
        raise RuntimeError("recovered inference runner SHA-256 mismatch")
    spec = importlib.util.spec_from_file_location("task005_recovered_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load recovered runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_arm(
    module: ModuleType,
    base_write_json: Any,
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    entries: list[dict[str, Any]],
    model_path: Path,
    model_sha256: str,
    adapter_sha256: str | None,
    checkpoint_identifier: str,
    model_revision: str,
    warmup_audio: Path,
    output_dir: Path,
    fine_tuned: bool,
) -> None:
    module.BENCHMARK_ID = manifest["dataset_id"]
    module.BENCHMARK_VERSION = manifest["dataset_version"]
    module.benchmark_entries = lambda _: (manifest, entries)

    def write_json(path: Path, payload: dict[str, Any]) -> None:
        if path.name == "inference_summary.json":
            payload["expected_utterances"] = len(entries)
        base_write_json(path, payload)

    # ponytail: reuse the checksum-pinned historical runner until NeMo's API contract changes.
    module.write_json = write_json
    arm = "fine-tuned" if fine_tuned else "pretrained"
    args = Namespace(
        benchmark_manifest=manifest_path,
        benchmark_package_sha256=sha256_file(manifest_path),
        checkpoint_identifier=checkpoint_identifier,
        expected_adapter_sha256=adapter_sha256,
        expected_model_sha256=model_sha256,
        experiment_id=f"gate2-{manifest['dataset_id']}-{arm}-v0",
        fine_tuned=fine_tuned,
        model=model_path,
        model_revision=model_revision,
        output_dir=output_dir,
        smoke_audio=warmup_audio,
    )
    module.run_benchmark(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovered-runner", type=Path, required=True)
    parser.add_argument("--independent-manifest", type=Path, required=True)
    parser.add_argument("--standard-manifest", type=Path, required=True)
    parser.add_argument("--pretrained-model", type=Path)
    parser.add_argument("--pretrained-sha256")
    parser.add_argument("--pretrained-identifier")
    parser.add_argument("--fine-tuned-model", type=Path)
    parser.add_argument("--fine-tuned-sha256")
    parser.add_argument("--adapter-sha256")
    parser.add_argument("--fine-tuned-identifier")
    parser.add_argument("--model-revision")
    parser.add_argument("--warmup-audio", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-manifests-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = {
        "independent-busan": (args.independent_manifest, *read_dataset(args.independent_manifest)),
        "standard-korean": (args.standard_manifest, *read_dataset(args.standard_manifest)),
    }
    if args.validate_manifests_only:
        print(
            json.dumps(
                {
                    name: {
                        "dataset_id": manifest["dataset_id"],
                        "utterances": len(entries),
                    }
                    for name, (_, manifest, entries) in datasets.items()
                },
                indent=2,
            )
        )
        return 0

    required = (
        "pretrained_model",
        "pretrained_sha256",
        "pretrained_identifier",
        "fine_tuned_model",
        "fine_tuned_sha256",
        "adapter_sha256",
        "fine_tuned_identifier",
        "model_revision",
        "warmup_audio",
        "output_dir",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise RuntimeError(f"missing inference arguments: {', '.join(missing)}")
    module = load_runner(args.recovered_runner)
    base_write_json = module.write_json
    for dataset_name, (path, manifest, entries) in datasets.items():
        for fine_tuned in (False, True):
            run_arm(
                module,
                base_write_json,
                manifest_path=path,
                manifest=manifest,
                entries=entries,
                model_path=args.fine_tuned_model if fine_tuned else args.pretrained_model,
                model_sha256=args.fine_tuned_sha256 if fine_tuned else args.pretrained_sha256,
                adapter_sha256=args.adapter_sha256 if fine_tuned else None,
                checkpoint_identifier=(
                    args.fine_tuned_identifier if fine_tuned else args.pretrained_identifier
                ),
                model_revision=args.model_revision,
                warmup_audio=args.warmup_audio,
                output_dir=args.output_dir / dataset_name / (
                    "fine-tuned" if fine_tuned else "pretrained"
                ),
                fine_tuned=fine_tuned,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
