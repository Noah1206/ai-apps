#!/usr/bin/env python3
"""Exercise the real ASR backend through the FastAPI request contract once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastapi.testclient import TestClient

from busan_lab.speech_api import SpeechAPISettings, create_speech_app
from busan_lab.speech_runtime import (
    DEFAULT_ADAPTER_SHA256,
    DEFAULT_MODEL_PATH,
    DEFAULT_MODEL_SHA256,
    DEFAULT_MODEL_VERSION,
    NemoOfflineASRBackend,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--target-text", required=True)
    parser.add_argument("--sentence-id", default="release-smoke-001")
    parser.add_argument("--focus-expression")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-commit", default="unknown")
    args = parser.parse_args()

    if not args.audio.is_file():
        parser.error(f"audio file does not exist: {args.audio}")
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")

    backend = NemoOfflineASRBackend(
        model_path=args.model,
        expected_model_sha256=DEFAULT_MODEL_SHA256,
        expected_adapter_sha256=DEFAULT_ADAPTER_SHA256,
        model_version=DEFAULT_MODEL_VERSION,
    )
    settings = SpeechAPISettings(
        service_version="speech-api-dev-20260816",
        git_commit=args.git_commit,
        eager_load=True,
    )
    application = create_speech_app(asr_backend=backend, settings=settings)
    with TestClient(application) as client:
        health = client.get("/health")
        version = client.get("/version")
        data = {
            "sentence_id": args.sentence_id,
            "target_text": args.target_text,
        }
        if args.focus_expression:
            data["focus_expression"] = args.focus_expression
        with args.audio.open("rb") as stream:
            attempt = client.post(
                "/v1/practice/attempt",
                files={"audio": (args.audio.name, stream, "audio/wav")},
                data=data,
                headers={"X-Request-ID": "release-smoke-20260816"},
            )
    if health.status_code != 200 or version.status_code != 200 or attempt.status_code != 200:
        raise RuntimeError(
            "Speech API smoke failed: "
            f"health={health.status_code}, version={version.status_code}, "
            f"attempt={attempt.status_code} {attempt.text}"
        )
    payload = {
        "status": "passed",
        "health": health.json(),
        "version": version.json(),
        "attempt": attempt.json(),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
