from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class LLMCollectionRunConfig:
    planner_type: str
    model: str
    environment_config_path: str
    behavior_set_path: str
    goal: str
    task_prompt: str
    system_prompt_path: str
    max_tree_ticks: int
    output_dir: str


class RunConfigError(ValueError):
    """Raised when an LLM collection run config is invalid."""


def load_llm_collection_run_config(path: str) -> LLMCollectionRunConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    return LLMCollectionRunConfig(
        planner_type=str(_require(raw, "planner_type")),
        model=str(_require(raw, "model")),
        environment_config_path=_resolve_path(config_path, _require(raw, "environment_config_path")),
        behavior_set_path=_resolve_path(config_path, _require(raw, "behavior_set_path")),
        goal=str(_require(raw, "goal")),
        task_prompt=str(_require(raw, "task_prompt")),
        system_prompt_path=_resolve_path(config_path, _require(raw, "system_prompt_path")),
        max_tree_ticks=int(_require(raw, "max_tree_ticks")),
        output_dir=_resolve_path(config_path, _require(raw, "output_dir")),
    )


def _require(raw_config: dict[str, object], key: str) -> object:
    if key not in raw_config:
        raise RunConfigError(f"Missing required run config field '{key}'.")
    return raw_config[key]


def _resolve_path(config_path: Path, raw_path: object) -> str:
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((config_path.parent / candidate).resolve())
