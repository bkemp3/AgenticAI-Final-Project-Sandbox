import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic.behaviors import load_behavior_catalog
from agentic.bt_spec.nodes import LeafNode, SequenceNode
from agentic.bt_spec.tree_structure import BehaviorTreeStructure
from agentic.orchestration.graph import build_orchestration_app
from agentic.planning.base import BasePlanner
from agentic.runtime_feedback import PlannerRepairRequest, RuntimeCriticAction, RuntimeCriticDecision
from agentic.tasks import load_task_adapter


class ScriptedRepairPlanner(BasePlanner):
    """Deterministic planner used to demonstrate runtime repair routing."""

    def create_plan(self, goal: str) -> BehaviorTreeStructure:
        return BehaviorTreeStructure(
            goal=goal,
            description="Semantically bad initial tree that picks up an object but never deposits it.",
            root=SequenceNode(
                type="sequence",
                name="bad_pickup_only_plan",
                children=[
                    LeafNode(type="sense_collection_world"),
                    LeafNode(type="select_best_material"),
                    LeafNode(type="navigate_to_selected_object"),
                    LeafNode(type="attempt_pickup_selected_object"),
                ],
            ),
        )

    def repair_plan(
        self,
        goal: str,
        current_tree: BehaviorTreeStructure,
        repair_request: PlannerRepairRequest,
    ) -> BehaviorTreeStructure:
        return BehaviorTreeStructure(
            goal=goal,
            description="Repaired tree that deposits the held object before checking value-goal success.",
            root=SequenceNode(
                type="sequence",
                name="deposit_after_pickup_repair",
                children=[
                    LeafNode(type="holding_object"),
                    LeafNode(type="navigate_to_dropoff"),
                    LeafNode(type="deposit_held_object"),
                    LeafNode(type="collection_value_goal_met", params=[{"key": "target_value", "value": 11}]),
                ],
            ),
        )


class ScriptedRuntimeCritic:
    """Generic scripted critic: repair if BT says SUCCESS but task progress says objective is not done."""

    def assess(self, **context) -> RuntimeCriticDecision:
        task_progress = context.get("task_progress_signals") or {}
        bt_status = context.get("bt_status")
        task_success = bool(task_progress.get("task_success"))
        if bt_status == "SUCCESS" and not task_success:
            return RuntimeCriticDecision(
                action=RuntimeCriticAction.REQUEST_REPAIR,
                diagnosis="The tree terminated successfully, but task progress signals show the objective is still unsatisfied.",
                repair_instructions=[
                    "Preserve the current world state.",
                    "Ensure the plan completes the objective instead of stopping after an intermediate action.",
                    "Prefer a repair focused on the subtree that ended early.",
                ],
                confidence=0.98,
            )
        return RuntimeCriticDecision(
            action=RuntimeCriticAction.CONTINUE,
            diagnosis="Execution remains aligned with the current task signals.",
            confidence=0.9,
        )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    task_adapter = load_task_adapter("agentic.tasks.collection.adapter")
    behavior_catalog = load_behavior_catalog(str(repo_root / "behavior_sets" / "collection.yaml"))
    world = task_adapter.load_world(str(repo_root / "configs" / "collection_env_runtime_repair_demo.yaml"))
    app = build_orchestration_app(include_runtime_critic=True)

    final_state = app.invoke(
        {
            "goal": "maximize_collection_value",
            "model": None,
            "task_prompt": (
                "Deposit enough visible material value to satisfy the task objective. "
                "Picking up an object without depositing it is not sufficient."
            ),
            "planner_type": "llm",
            "world_state": world,
            "behavior_catalog": behavior_catalog,
            "task_adapter": task_adapter,
            "system_prompt_override": None,
            "user_prompt_override": None,
            "max_tree_ticks": 10,
            "retry_limit": 0,
            "critic_enabled": True,
            "critic_model": None,
            "critic_interval_ticks": 4,
            "critic_max_repairs": 1,
            "critic_context_window_events": 6,
            "critic_context_window_ticks": 6,
            "tree_output_dir": str(repo_root / "outputs" / "trees"),
            "planner": ScriptedRepairPlanner(),
            "runtime_critic": ScriptedRuntimeCritic(),
            "tree_spec": None,
            "compiled_tree": None,
            "execution_status": None,
            "runtime_bt_status": None,
            "error_message": None,
            "tree_image_path": None,
            "graph_mermaid_path": None,
            "graph_image_path": None,
            "tick_count": 0,
            "repair_count": 0,
            "runtime_tick_trace": [],
            "critic_decision": None,
            "critic_history": [],
            "planner_repair_request": None,
            "critic_due": False,
            "execution_should_continue": False,
        }
    )

    print(f"Execution status: {final_state['execution_status']}")
    print(f"Repairs used: {final_state['repair_count']}")
    print("Critic history:")
    for item in final_state.get("critic_history") or []:
        print(f"  {item}")
    print("Final tree:")
    print(final_state["tree_spec"].model_dump_json(indent=2))
    print(f"Final metrics: {task_adapter.get_metrics(world)}")


if __name__ == "__main__":
    main()
