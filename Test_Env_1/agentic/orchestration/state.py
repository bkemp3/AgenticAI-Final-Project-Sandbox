from __future__ import annotations

from typing import TypedDict

from agentic.bt_spec.tree_structure import BehaviorTreeStructure
from agentic.planning.base import BasePlanner
from agentic.world_state import WorldState


class OrchestrationState(TypedDict):
    """Shared LangGraph state for the end-to-end planning pipeline."""

    goal: str
    planner_type: str
    world_state: WorldState
    planner: BasePlanner | None
    tree_spec: BehaviorTreeStructure | None
    compiled_tree: object | None
    execution_status: str | None
    error_message: str | None
    tree_image_path: str | None
    graph_mermaid_path: str | None
    graph_image_path: str | None
