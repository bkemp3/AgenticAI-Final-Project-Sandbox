from __future__ import annotations

from typing import TypedDict

from agentic.behaviors.catalog import BehaviorCatalog
from agentic.bt_spec.tree_structure import BehaviorTreeStructure
from agentic.planning.base import BasePlanner


class OrchestrationState(TypedDict):
    """Shared LangGraph state for the end-to-end planning pipeline."""

    goal: str
    planner_type: str
    world_state: object
    behavior_catalog: BehaviorCatalog
    task_adapter: object | None
    system_prompt_override: str | None
    user_prompt_override: str | None
    max_tree_ticks: int | None
    retry_limit: int | None
    tree_output_dir: str | None
    planner: BasePlanner | None
    tree_spec: BehaviorTreeStructure | None
    compiled_tree: object | None
    execution_status: str | None
    error_message: str | None
    tree_image_path: str | None
    graph_mermaid_path: str | None
    graph_image_path: str | None
