from __future__ import annotations

from agentic.behaviors.catalog import BehaviorCatalog
from agentic.bt_spec.tree_structure import BehaviorTreeStructure
from agentic.planning.prompting import build_behavior_catalog_block, build_behavior_subset_block
from agentic.serialization import to_pretty_json


def build_runtime_critic_system_prompt(catalog: BehaviorCatalog) -> str:
    return "\n\n".join(
        [
            (
                "You are a runtime critic for a behavior-tree task runner. "
                "Assess whether execution is still aligned with the task objective."
            ),
            (
                "Do not rewrite the tree yourself. Your only job is to decide whether execution should continue "
                "or whether the planner should repair the tree."
            ),
            (
                "Distinguish behavior-tree control status from task success. "
                "A BT status of SUCCESS does not by itself mean the task objective is satisfied."
            ),
            (
                "Request repair when the execution looks stalled, semantically misaligned, prematurely terminal, "
                "or likely to fail the objective without intervention."
            ),
            (
                "Use only the provided visible observation, recent events, recent tick trace, task progress signals, "
                "metrics, and behavior descriptions."
            ),
            build_behavior_catalog_block(catalog),
        ]
    ).strip()


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
    sections = [
        f"Goal:\n{goal}",
        f"Task objective:\n{task_objective.strip()}",
        f"Current behavior tree:\n{tree_spec.model_dump_json(indent=2)}",
        f"Current BT status:\n{bt_status or 'UNKNOWN'}",
        f"Total ticks executed:\n{total_ticks}",
        f"Repairs already used:\n{repair_count}",
        f"Visible world observation:\n{to_pretty_json(visible_world_observation)}",
        f"Recent event window:\n{to_pretty_json(recent_events)}",
        f"Recent tick/status trace:\n{to_pretty_json(recent_tick_trace)}",
        f"Task progress and success signals:\n{to_pretty_json(task_progress_signals)}",
        f"Current metrics:\n{to_pretty_json(metrics)}",
        used_behavior_block,
        (
            "Return CONTINUE only if execution appears meaningfully aligned with the task objective. "
            "Return REQUEST_REPAIR if the tree is stalled, semantically off-course, or terminal without objective success."
        ),
    ]
    return "\n\n".join(sections).strip()


def build_used_behavior_block(catalog: BehaviorCatalog, tree_spec: BehaviorTreeStructure) -> str:
    return "Behavior descriptions for nodes used in the current tree:\n" + build_behavior_subset_block(
        catalog,
        tree_spec,
    )
