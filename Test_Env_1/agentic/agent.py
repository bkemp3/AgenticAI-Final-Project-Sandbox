from agentic.behaviors.catalog import load_behavior_catalog
from agentic.orchestration.graph import build_orchestration_app
from agentic.orchestration.state import OrchestrationState
from agentic.orchestration.visualization import export_langgraph_visualization
class Agent:
    """Agent that runs the pipeline through LangGraph orchestration."""

    def __init__(
        self,
        world_state: object,
        planner_type: str = "rule_based",
        behavior_set_path: str | None = None,
    ) -> None:
        self.world_state = world_state
        self.planner_type = planner_type
        self.behavior_catalog = load_behavior_catalog(behavior_set_path)
        self.app = build_orchestration_app()

    def run(self, goal: str) -> OrchestrationState:
        """Run the full planning and execution workflow for a goal."""

        graph_artifacts = export_langgraph_visualization(
            self.app,
            name="langgraph_pipeline",
        )
        initial_state: OrchestrationState = {
            "goal": goal,
            "planner_type": self.planner_type,
            "world_state": self.world_state,
            "behavior_catalog": self.behavior_catalog,
            "task_adapter": None,
            "system_prompt_override": None,
            "user_prompt_override": None,
            "max_tree_ticks": None,
            "retry_limit": None,
            "tree_output_dir": None,
            "planner": None,
            "tree_spec": None,
            "compiled_tree": None,
            "execution_status": None,
            "error_message": None,
            "tree_image_path": None,
            "graph_mermaid_path": graph_artifacts["mermaid"],
            "graph_image_path": graph_artifacts["png"],
        }
        return self.app.invoke(initial_state)
