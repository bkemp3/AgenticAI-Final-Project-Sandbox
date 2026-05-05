from __future__ import annotations

from typing import TypedDict

from agentic.behaviors.catalog import BehaviorCatalog
from agentic.bt_spec.tree_structure import BehaviorTreeStructure
from agentic.planning.base import BasePlanner
from agentic.runtime_feedback import PlannerRepairRequest, RuntimeCriticDecision


class OrchestrationState(TypedDict):
    """Shared LangGraph state for the end-to-end planning pipeline."""

    goal: str
    model: str | None
    task_prompt: str | None
    planner_type: str
    world_state: object
    behavior_catalog: BehaviorCatalog
    task_adapter: object | None
    system_prompt_override: str | None
    user_prompt_override: str | None
    max_tree_ticks: int | None
    retry_limit: int | None
    critic_enabled: bool
    critic_model: str | None
    critic_interval_ticks: int | None
    critic_max_repairs: int | None
    critic_context_window_events: int | None
    critic_context_window_ticks: int | None
    tree_output_dir: str | None
    planner: BasePlanner | None
    runtime_critic: object | None
    tree_spec: BehaviorTreeStructure | None
    compiled_tree: object | None
    execution_status: str | None
    runtime_bt_status: str | None
    error_message: str | None
    tree_image_path: str | None
    tree_spec_history: list[dict[str, object]]
    tree_artifact_history: list[dict[str, object]]
    graph_mermaid_path: str | None
    graph_image_path: str | None
    tick_count: int
    repair_count: int
    runtime_tick_trace: list[dict[str, object]]
    critic_decision: RuntimeCriticDecision | None
    critic_history: list[dict[str, object]]
    planner_repair_request: PlannerRepairRequest | None
    critic_due: bool
    execution_should_continue: bool
