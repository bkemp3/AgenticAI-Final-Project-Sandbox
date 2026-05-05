from __future__ import annotations

from pathlib import Path

from agentic.behaviors.catalog import BehaviorCatalog
from agentic.bt_spec.tree_structure import BehaviorTreeStructure
from agentic.planning.prompting import build_behavior_catalog_block, build_behavior_subset_block
from agentic.serialization import to_pretty_json

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
RUNTIME_CRITIC_SYSTEM_PROMPT_PATH = PROMPTS_DIR / "runtime_critic_system_base.txt"
RUNTIME_CRITIC_USER_PROMPT_PATH = PROMPTS_DIR / "runtime_critic_user_base.txt"


def build_runtime_critic_system_prompt(catalog: BehaviorCatalog) -> str:
    base = _load_prompt_asset(RUNTIME_CRITIC_SYSTEM_PROMPT_PATH)
    return "\n\n".join([base, build_behavior_catalog_block(catalog)]).strip()


def build_runtime_critic_user_prompt(
    *,
    goal: str,
    task_objective: str,
    tree_spec: BehaviorTreeStructure,
    visible_world_observation: object | None,
    recent_events: list[object],
    recent_tick_trace: list[object],
    task_progress_signals: object | None,
    metrics: object | None,
    used_behavior_block: str,
    total_ticks: int,
    repair_count: int,
    bt_status: str | None,
) -> str:
    return _load_prompt_asset(RUNTIME_CRITIC_USER_PROMPT_PATH).format(
        goal=goal,
        task_objective=task_objective.strip(),
        tree_json=tree_spec.model_dump_json(indent=2),
        bt_status=bt_status or "UNKNOWN",
        total_ticks=total_ticks,
        repair_count=repair_count,
        visible_world_observation=to_pretty_json(visible_world_observation),
        recent_events=to_pretty_json(recent_events),
        recent_tick_trace=to_pretty_json(recent_tick_trace),
        task_progress_signals=to_pretty_json(task_progress_signals),
        metrics=to_pretty_json(metrics),
        used_behavior_block=used_behavior_block,
    ).strip()


def build_used_behavior_block(catalog: BehaviorCatalog, tree_spec: BehaviorTreeStructure) -> str:
    return "Behavior descriptions for nodes used in the current tree:\n" + build_behavior_subset_block(
        catalog,
        tree_spec,
    )


def _load_prompt_asset(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()
