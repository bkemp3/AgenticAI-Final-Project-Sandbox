from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CriticRunConfig:
    enabled: bool
    model: str | None
    interval_ticks: int
    max_repairs: int
    context_window_events: int
    context_window_ticks: int


@dataclass(frozen=True)
class LLMTaskRunConfig:
    planner_type: str
    model: str
    task_adapter: str
    environment_config_path: str
    behavior_set_path: str
    goal: str
    task_prompt: str
    system_prompt_path: str
    user_prompt_include_environment_summary: bool
    max_tree_ticks: int
    retry_limit: int
    output_dir: str
    output_namespace: str | None
    critic: CriticRunConfig


class RunConfigError(ValueError):
    """Raised when an LLM collection run config is invalid."""


def load_llm_task_run_config(path: str) -> LLMTaskRunConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    critic_raw = raw.get("critic") or {}

    return LLMTaskRunConfig(
        planner_type=str(_require(raw, "planner_type")),
        model=str(_require(raw, "model")),
        task_adapter=str(_require(raw, "task_adapter")),
        environment_config_path=_resolve_path(config_path, _require(raw, "environment_config_path")),
        behavior_set_path=_resolve_path(config_path, _require(raw, "behavior_set_path")),
        goal=str(_require(raw, "goal")),
        task_prompt=str(_require(raw, "task_prompt")),
        system_prompt_path=_resolve_path(config_path, _require(raw, "system_prompt_path")),
        user_prompt_include_environment_summary=bool(raw.get("user_prompt_include_environment_summary", True)),
        max_tree_ticks=int(_require(raw, "max_tree_ticks")),
        retry_limit=int(raw.get("retry_limit", 1)),
        output_dir=_resolve_path(config_path, _require(raw, "output_dir")),
        output_namespace=str(raw["output_namespace"]).strip() if raw.get("output_namespace") is not None else None,
        critic=CriticRunConfig(
            enabled=bool(critic_raw.get("enabled", False)),
            model=str(critic_raw["model"]) if critic_raw.get("model") is not None else None,
            interval_ticks=int(critic_raw.get("interval_ticks", 3)),
            max_repairs=int(critic_raw.get("max_repairs", 1)),
            context_window_events=int(critic_raw.get("context_window_events", 10)),
            context_window_ticks=int(critic_raw.get("context_window_ticks", 6)),
        ),
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
