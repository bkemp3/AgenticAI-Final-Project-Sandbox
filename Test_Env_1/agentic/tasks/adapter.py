from __future__ import annotations

import importlib
from typing import Protocol

import py_trees

from agentic.bt_spec.tree_structure import BehaviorTreeStructure


class TaskAdapterProtocol(Protocol):
    def load_world(self, environment_config_path: str) -> object: ...

    def summarize_world(self, world: object) -> str | None: ...

    def describe_tick(self, world: object, tree: py_trees.trees.BehaviourTree, tick: int) -> str: ...

    def render_tick(self, world: object, tree: py_trees.trees.BehaviourTree, tick: int) -> str | None: ...

    def get_visible_observation(self, world: object, tree: py_trees.trees.BehaviourTree) -> object | None: ...

    def get_events(self, world: object) -> list[object]: ...

    def get_metrics(self, world: object) -> object: ...

    def get_progress_signals(self, world: object) -> object: ...

    def is_success(self, world: object) -> bool: ...

    def is_terminal(self, world: object) -> bool: ...

    def validate_plan(self, tree_spec: BehaviorTreeStructure) -> str | None: ...

def load_task_adapter(import_path: str) -> TaskAdapterProtocol:
    module_path, _, attr_name = import_path.rpartition(".")
    if not module_path or not attr_name:
        raise ValueError(
            f"Invalid task_adapter path: {import_path}. "
            "Use a full import path like 'agentic.tasks.collection.adapter'."
        )

    module = importlib.import_module(module_path)
    adapter = getattr(module, attr_name, None)
    if adapter is None:
        raise ValueError(f"Task adapter not found: {import_path}")

    required_methods = (
        "load_world",
        "summarize_world",
        "describe_tick",
        "render_tick",
        "get_visible_observation",
        "get_events",
        "get_metrics",
        "get_progress_signals",
        "is_success",
        "is_terminal",
        "validate_plan",
    )
    missing_methods = [method for method in required_methods if not callable(getattr(adapter, method, None))]
    if missing_methods:
        missing = ", ".join(sorted(missing_methods))
        raise TypeError(f"Task adapter '{import_path}' is missing required callables: {missing}")

    return adapter
