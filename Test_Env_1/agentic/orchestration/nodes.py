from __future__ import annotations

from pathlib import Path

from agentic.bt_runtime.compiler import compile_behavior_tree
from agentic.bt_runtime.executor import execute_tree_batch
from agentic.bt_runtime.visualization import export_tree_image, print_ascii_tree
from agentic.orchestration.state import OrchestrationState
from agentic.planning.llm_planner import LLMPlanner
from agentic.planning.rule_based_planner import RuleBasedPlanner
from agentic.runtime_critic import LLMRuntimeCritic
from agentic.runtime_feedback import PlannerRepairRequest, RuntimeCriticAction
from agentic.serialization import serialize_value


def select_planner(state: OrchestrationState) -> dict[str, object]:
    """Select the planner implementation for the run."""

    if state.get("planner") is not None:
        return {
            "planner": state["planner"],
            "planner_type": _normalize_planner_type(state["planner_type"]),
            "runtime_critic": state.get("runtime_critic"),
        }

    planner_type = _normalize_planner_type(state["planner_type"])
    if planner_type == "llm":
        task_adapter = state.get("task_adapter")
        plan_validator = getattr(task_adapter, "validate_plan", None) if task_adapter is not None else None
        planner = LLMPlanner(
            behavior_catalog=state["behavior_catalog"],
            model=state.get("model") or "gpt-4.1-mini",
            system_prompt_override=state.get("system_prompt_override"),
            user_prompt_override=state.get("user_prompt_override"),
            retry_limit=state.get("retry_limit") or 0,
            plan_validator=plan_validator,
        )
        runtime_critic = None
        if state.get("critic_enabled"):
            runtime_critic = LLMRuntimeCritic(
                behavior_catalog=state["behavior_catalog"],
                model=state.get("critic_model") or state.get("model") or "gpt-4.1-mini",
            )
    else:
        if state.get("critic_enabled"):
            return _error_update("Runtime critic repair currently requires the LLM planner.")
        planner = RuleBasedPlanner(behavior_catalog=state["behavior_catalog"])
        runtime_critic = None
    return {"planner": planner, "planner_type": planner_type, "runtime_critic": runtime_critic}


def generate_plan(state: OrchestrationState) -> dict[str, object]:
    """Generate a validated behavior tree structure."""

    try:
        planner = state["planner"]
        if planner is None:
            raise ValueError("Planner was not selected before plan generation.")
        repair_request = state.get("planner_repair_request")
        current_tree = state.get("tree_spec")
        if repair_request is not None:
            if current_tree is None:
                raise ValueError("Repair was requested but no current tree is available.")
            tree_spec = planner.repair_plan(state["goal"], current_tree, repair_request)
        else:
            tree_spec = planner.create_plan(state["goal"])
        spec_path = _persist_tree_spec(
            tree_spec=tree_spec,
            output_dir=state.get("tree_output_dir"),
            repair_count=state.get("repair_count") or 0,
        )
        tree_spec_history = list(state.get("tree_spec_history") or [])
        tree_spec_history.append(
            {
                "repair_count": state.get("repair_count") or 0,
                "goal": tree_spec.goal,
                "description": tree_spec.description,
                "path": spec_path,
            }
        )
        return {
            "tree_spec": tree_spec,
            "tree_spec_history": tree_spec_history,
            "planner_repair_request": None,
            "critic_due": False,
            "critic_decision": None,
        }
    except Exception as exc:
        return _error_update(f"Plan generation failed: {exc}")


def compile_tree(state: OrchestrationState) -> dict[str, object]:
    """Compile the validated spec into a py_trees runtime tree."""

    try:
        tree_spec = state["tree_spec"]
        if tree_spec is None:
            raise ValueError("No behavior tree structure is available to compile.")
        compiled_tree = compile_behavior_tree(
            tree_spec,
            state["world_state"],
            state["behavior_catalog"].runtime_registry(),
        )
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
        repair_count = state.get("repair_count") or 0
        goal_name = state["tree_spec"].goal if state["tree_spec"] is not None else "behavior_tree"
        artifact_name = goal_name if repair_count == 0 else f"{goal_name}_repair_{repair_count}"
        artifacts = export_tree_image(
            compiled_tree,
            name=artifact_name,
            output_dir=state.get("tree_output_dir") or "outputs/trees",
        )
        tree_image_path = _pick_tree_image_path(artifacts)
        tree_artifact_history = list(state.get("tree_artifact_history") or [])
        tree_artifact_history.append(
            {
                "repair_count": repair_count,
                "goal": goal_name,
                "artifact_name": artifact_name,
                "artifacts": artifacts or {},
                "primary_path": tree_image_path,
            }
        )
        return {"tree_image_path": tree_image_path, "tree_artifact_history": tree_artifact_history}
    except Exception as exc:
        return _error_update(f"Tree visualization failed: {exc}")


def execute_tree_node(state: OrchestrationState) -> dict[str, object]:
    """Execute the compiled tree for one runtime batch."""

    try:
        compiled_tree = state["compiled_tree"]
        if compiled_tree is None:
            raise ValueError("No compiled tree is available to execute.")
        task_adapter = state.get("task_adapter")

        current_tick = state.get("tick_count") or 0
        max_tree_ticks = state.get("max_tree_ticks") or 10
        remaining_ticks = max_tree_ticks - current_tick
        if remaining_ticks <= 0:
            return {
                "execution_status": "MAX_TICKS_EXCEEDED",
                "execution_should_continue": False,
                "critic_due": False,
            }

        if task_adapter is None:
            for _ in range(remaining_ticks):
                compiled_tree.tick()
                current_tick += 1
                status = compiled_tree.root.status
                if status.name in {"SUCCESS", "FAILURE"}:
                    return {
                        "tick_count": current_tick,
                        "runtime_bt_status": status.name,
                        "execution_status": status.name,
                        "execution_should_continue": False,
                        "critic_due": False,
                    }
            return {
                "tick_count": current_tick,
                "runtime_bt_status": compiled_tree.root.status.name,
                "execution_status": "MAX_TICKS_EXCEEDED",
                "execution_should_continue": False,
                "critic_due": False,
            }

        batch_limit = remaining_ticks
        if state.get("critic_enabled"):
            batch_limit = min(batch_limit, max(1, state.get("critic_interval_ticks") or 1))

        result = execute_tree_batch(
            compiled_tree,
            max_ticks=batch_limit,
            start_tick=current_tick,
            world_state=state["world_state"],
            task_adapter=task_adapter,
        )
        total_ticks = current_tick + result.ticks_executed
        full_trace = list(state.get("runtime_tick_trace") or [])
        full_trace.extend(serialize_value(entry) for entry in result.tick_trace)

        if result.task_success:
            return {
                "tick_count": total_ticks,
                "runtime_bt_status": result.bt_status,
                "runtime_tick_trace": full_trace,
                "execution_status": "SUCCESS",
                "execution_should_continue": False,
                "critic_due": False,
            }
        if result.task_terminal:
            return {
                "tick_count": total_ticks,
                "runtime_bt_status": result.bt_status,
                "runtime_tick_trace": full_trace,
                "execution_status": "TASK_TERMINAL",
                "execution_should_continue": False,
                "critic_due": False,
            }
        if total_ticks >= max_tree_ticks:
            return {
                "tick_count": total_ticks,
                "runtime_bt_status": result.bt_status,
                "runtime_tick_trace": full_trace,
                "execution_status": "MAX_TICKS_EXCEEDED",
                "execution_should_continue": False,
                "critic_due": False,
            }

        critic_due = bool(state.get("critic_enabled")) and (result.bt_terminal or result.ticks_executed >= batch_limit)
        if result.bt_terminal and not critic_due:
            return {
                "tick_count": total_ticks,
                "runtime_bt_status": result.bt_status,
                "runtime_tick_trace": full_trace,
                "execution_status": result.bt_status,
                "execution_should_continue": False,
                "critic_due": False,
            }

        return {
            "tick_count": total_ticks,
            "runtime_bt_status": result.bt_status,
            "runtime_tick_trace": full_trace,
            "execution_status": None,
            "execution_should_continue": not result.bt_terminal,
            "critic_due": critic_due,
        }
    except Exception as exc:
        return _error_update(f"Tree execution failed: {exc}")


def runtime_critic_node(state: OrchestrationState) -> dict[str, object]:
    """Ask the runtime critic whether execution should continue or be repaired."""

    try:
        runtime_critic = state.get("runtime_critic")
        if runtime_critic is None:
            raise ValueError("Runtime critic is not configured.")
        tree_spec = state.get("tree_spec")
        compiled_tree = state.get("compiled_tree")
        task_adapter = state.get("task_adapter")
        if tree_spec is None or compiled_tree is None or task_adapter is None:
            raise ValueError("Runtime critic requires a current tree, compiled tree, and task adapter.")

        recent_events = task_adapter.get_events(state["world_state"])[-(state.get("critic_context_window_events") or 10):]
        recent_tick_trace = (state.get("runtime_tick_trace") or [])[-(state.get("critic_context_window_ticks") or 6):]
        visible_world_observation = task_adapter.get_visible_observation(state["world_state"], compiled_tree)
        task_progress_signals = task_adapter.get_progress_signals(state["world_state"])
        metrics = task_adapter.get_metrics(state["world_state"])
        decision = runtime_critic.assess(
            goal=state["goal"],
            task_objective=state.get("task_prompt") or state["goal"],
            tree_spec=tree_spec,
            visible_world_observation=visible_world_observation,
            recent_events=recent_events,
            recent_tick_trace=recent_tick_trace,
            task_progress_signals=task_progress_signals,
            metrics=metrics,
            total_ticks=state.get("tick_count") or 0,
            repair_count=state.get("repair_count") or 0,
            bt_status=state.get("runtime_bt_status"),
        )

        critic_history = list(state.get("critic_history") or [])
        critic_history.append(serialize_value(decision))
        updates: dict[str, object] = {
            "critic_decision": decision,
            "critic_history": critic_history,
            "critic_due": False,
        }
        if decision.action == RuntimeCriticAction.REQUEST_REPAIR:
            repairs_used = state.get("repair_count") or 0
            repair_limit = state.get("critic_max_repairs") or 0
            if repairs_used >= repair_limit:
                updates["execution_status"] = "REPAIR_LIMIT_REACHED"
                updates["execution_should_continue"] = False
                return updates
            updates["planner_repair_request"] = PlannerRepairRequest(
                diagnosis=decision.diagnosis,
                repair_instructions=decision.repair_instructions,
                suspected_node=decision.suspected_node,
                confidence=decision.confidence,
                visible_world_observation=visible_world_observation,
                recent_events=recent_events,
                recent_tick_trace=recent_tick_trace,
                task_progress_signals=task_progress_signals,
                metrics=metrics,
            )
            updates["repair_count"] = repairs_used + 1
            updates["execution_should_continue"] = False
            updates["execution_status"] = None
            return updates

        if state.get("runtime_bt_status") in {"SUCCESS", "FAILURE"}:
            updates["execution_status"] = state.get("runtime_bt_status")
            updates["execution_should_continue"] = False
        else:
            updates["execution_status"] = None
            updates["execution_should_continue"] = True
        return updates
    except Exception as exc:
        return _error_update(f"Runtime critic failed: {exc}")


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
    return {"error_message": message, "execution_should_continue": False, "critic_due": False}


def _persist_tree_spec(
    *,
    tree_spec,
    output_dir: str | None,
    repair_count: int,
) -> str | None:
    if output_dir is None:
        return None

    tree_dir = Path(output_dir)
    tree_dir.mkdir(parents=True, exist_ok=True)
    goal_name = tree_spec.goal
    filename = f"{goal_name}.json" if repair_count == 0 else f"{goal_name}_repair_{repair_count}.json"
    path = tree_dir / filename
    path.write_text(tree_spec.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return str(path)
