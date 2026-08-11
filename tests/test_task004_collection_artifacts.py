from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_PATH = (
    PROJECT_ROOT / "artifacts" / "task-004" / "collection-prompts-v0.jsonl"
)
SOLO_PROMPTS_PATH = (
    PROJECT_ROOT / "artifacts" / "task-004" / "SOLO_SPEAKER_300.md"
)
SOLO_PROMPT_PATTERN = re.compile(r"^- (T004-S\d{3}): (.+)$", re.MULTILINE)

FROZEN_BENCHMARK_SURFACES = {
    "와따 맛있노",
    "마, 괜찮다 아이가.",
    "내일 같이 가재이",
    "여기 좀 앉으이소.",
    "지금 뭐 하노?",
    "그거 아이다.",
    "오늘 와 이리 춥노?",
    "밥 묵었나?",
    "니 지금 어데고?",
    "국밥 하나 주이소",
}


def _surface_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _load_prompts() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in PROMPTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_task004_collection_prompts_are_unique_and_benchmark_disjoint() -> None:
    prompts = _load_prompts()

    assert len(prompts) == 50
    assert [prompt["prompt_id"] for prompt in prompts] == [
        f"T004-P{index:03d}" for index in range(1, 51)
    ]
    assert sum(prompt["recommended_split"] == "train" for prompt in prompts) == 35
    assert (
        sum(prompt["recommended_split"] == "validation" for prompt in prompts) == 15
    )

    surface_keys = [_surface_key(prompt["elicitation_text"]) for prompt in prompts]
    benchmark_keys = {_surface_key(value) for value in FROZEN_BENCHMARK_SURFACES}
    assert len(surface_keys) == len(set(surface_keys))
    assert set(surface_keys).isdisjoint(benchmark_keys)

    for prompt in prompts:
        assert set(prompt) == {
            "schema_version",
            "prompt_id",
            "recommended_split",
            "elicitation_text",
            "target_expressions",
            "scenario",
            "prompt_status",
            "approved_by",
        }
        assert prompt["schema_version"] == "1.0.0"
        assert prompt["prompt_status"] == "candidate"
        assert prompt["approved_by"] is None
        assert prompt["target_expressions"]


def test_single_speaker_train_prompts_have_300_unique_disjoint_sentences() -> None:
    source = SOLO_PROMPTS_PATH.read_text(encoding="utf-8")
    prompts = SOLO_PROMPT_PATTERN.findall(source)

    assert len(prompts) == 300
    assert [prompt_id for prompt_id, _text in prompts] == [
        f"T004-S{index:03d}" for index in range(1, 301)
    ]

    surface_keys = [_surface_key(text) for _prompt_id, text in prompts]
    benchmark_keys = {_surface_key(value) for value in FROZEN_BENCHMARK_SURFACES}
    elicitation_keys = {
        _surface_key(prompt["elicitation_text"]) for prompt in _load_prompts()
    }
    assert len(surface_keys) == len(set(surface_keys))
    assert set(surface_keys).isdisjoint(benchmark_keys)
    assert set(surface_keys).isdisjoint(elicitation_keys)
