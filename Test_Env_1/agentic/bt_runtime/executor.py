from __future__ import annotations

import py_trees
from pydantic import BaseModel, ConfigDict, Field

from agentic.runtime_feedback import RuntimeTickTraceEntry


class TreeExecutionBatchResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    ticks_executed: int
    bt_status: str | None
    bt_terminal: bool
    task_terminal: bool
    task_success: bool
    tick_trace: list[RuntimeTickTraceEntry] = Field(default_factory=list)


def execute_tree_batch(
    tree: py_trees.trees.BehaviourTree,
    *,
    max_ticks: int,
    start_tick: int,
    tree_generation: int,
    world_state: object,
    task_adapter: object,
) -> TreeExecutionBatchResult:
    """Tick a py_trees tree for a bounded batch and capture execution context."""

    tick_trace: list[RuntimeTickTraceEntry] = []
    last_status: py_trees.common.Status | None = None

    for offset in range(max_ticks):
        tree.tick()
        status = tree.root.status
        last_status = status
        tick_number = start_tick + offset + 1
        tick_trace.append(
            RuntimeTickTraceEntry(
                tick=tick_number,
                bt_status=status.name,
                summary=task_adapter.describe_tick(world_state, tree, tick_number),
                world_render=task_adapter.render_tick(world_state, tree, tick_number, tree_generation),
                active_nodes=_collect_active_nodes(tree.root),
            )
        )
        if task_adapter.is_terminal(world_state):
            return TreeExecutionBatchResult(
                ticks_executed=offset + 1,
                bt_status=status.name,
                bt_terminal=status in (py_trees.common.Status.SUCCESS, py_trees.common.Status.FAILURE),
                task_terminal=True,
                task_success=task_adapter.is_success(world_state),
                tick_trace=tick_trace,
            )
        if status in (py_trees.common.Status.SUCCESS, py_trees.common.Status.FAILURE):
            return TreeExecutionBatchResult(
                ticks_executed=offset + 1,
                bt_status=status.name,
                bt_terminal=True,
                task_terminal=False,
                task_success=task_adapter.is_success(world_state),
                tick_trace=tick_trace,
            )

    return TreeExecutionBatchResult(
        ticks_executed=max_ticks,
        bt_status=last_status.name if last_status is not None else None,
        bt_terminal=False,
        task_terminal=task_adapter.is_terminal(world_state),
        task_success=task_adapter.is_success(world_state),
        tick_trace=tick_trace,
    )


def _collect_active_nodes(node: py_trees.behaviour.Behaviour, path: str = "root") -> list[dict[str, str]]:
    status = getattr(node, "status", None)
    status_name = getattr(status, "name", "UNKNOWN")
    nodes: list[dict[str, str]] = []
    if path == "root" or status_name != py_trees.common.Status.INVALID.name:
        nodes.append(
            {
                "path": path,
                "name": node.name,
                "class": node.__class__.__name__,
                "status": status_name,
            }
        )
    for index, child in enumerate(getattr(node, "children", []) or []):
        nodes.extend(_collect_active_nodes(child, path=f"{path}.{index}"))
    return nodes
