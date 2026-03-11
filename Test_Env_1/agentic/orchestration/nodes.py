from __future__ import annotations

from pathlib import Path

from agentic.bt_runtime.compiler import compile_behavior_tree
from agentic.bt_runtime.executor import execute_tree
from agentic.bt_runtime.visualization import export_tree_image, print_ascii_tree
from agentic.orchestration.state import OrchestrationState
from agentic.planning.llm_planner import LLMPlanner
from agentic.planning.rule_based_planner import RuleBasedPlanner


def select_planner(state: OrchestrationState) -> dict[str, object]:
    """Select the planner implementation for the run."""

    planner_type = _normalize_planner_type(state["planner_type"])
    if planner_type == "llm":
        planner = LLMPlanner()
    else:
        planner = RuleBasedPlanner()
    return {"planner": planner, "planner_type": planner_type}


def generate_plan(state: OrchestrationState) -> dict[str, object]:
    """Generate a validated behavior tree structure."""

    try:
        planner = state["planner"]
        if planner is None:
            raise ValueError("Planner was not selected before plan generation.")
        tree_spec = planner.create_plan(state["goal"])
        return {"tree_spec": tree_spec}
    except Exception as exc:
        return _error_update(f"Plan generation failed: {exc}")


def compile_tree(state: OrchestrationState) -> dict[str, object]:
    """Compile the validated spec into a py_trees runtime tree."""

    try:
        tree_spec = state["tree_spec"]
        if tree_spec is None:
            raise ValueError("No behavior tree structure is available to compile.")
        compiled_tree = compile_behavior_tree(tree_spec, state["world_state"])
        return {"compiled_tree": compiled_tree}
    except Exception as exc:
        return _error_update(f"Tree compilation failed: {exc}")


def visualize_tree(state: OrchestrationState) -> dict[str, object]:
    """Print text visualization and attempt tree image export."""

    try:
        compiled_tree = state["compiled_tree"]
        if compiled_tree is None:
            raise ValueError("No compiled tree is available to visualize.")
        if state["tree_spec"] is None:
            raise ValueError("No behavior tree structure is available for naming outputs.")

        print("Runtime behavior tree:")
        print_ascii_tree(compiled_tree)
        artifacts = export_tree_image(compiled_tree, name=state["tree_spec"].goal)
        tree_image_path = _pick_tree_image_path(artifacts)
        return {"tree_image_path": tree_image_path}
    except Exception as exc:
        return _error_update(f"Tree visualization failed: {exc}")


def execute_tree_node(state: OrchestrationState) -> dict[str, object]:
    """Execute the compiled py_trees behavior tree."""

    try:
        compiled_tree = state["compiled_tree"]
        if compiled_tree is None:
            raise ValueError("No compiled tree is available to execute.")
        result = execute_tree(compiled_tree)
        return {"execution_status": result.name}
    except Exception as exc:
        return _error_update(f"Tree execution failed: {exc}")


def handle_error(state: OrchestrationState) -> dict[str, object]:
    """Record a readable error state for the pipeline."""

    error_message = state.get("error_message") or "Unknown orchestration error."
    print(f"Pipeline error: {error_message}")
    return {"execution_status": "ERROR", "error_message": error_message}


def _normalize_planner_type(planner_type: str) -> str:
    normalized = planner_type.strip().lower()
    if normalized in {"llm", "openai"}:
        return "llm"
    return "rule_based"


def _pick_tree_image_path(artifacts: dict[str, str] | None) -> str | None:
    if not artifacts:
        return None

    for key in ("svg", "png", "dot"):
        candidate = artifacts.get(key)
        if candidate and Path(candidate).exists():
            return candidate
    return None


def _error_update(message: str) -> dict[str, object]:
    return {"error_message": message}
