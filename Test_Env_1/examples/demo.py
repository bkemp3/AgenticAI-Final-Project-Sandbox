import argparse
from pathlib import Path
import sys

# Allow running the demo directly from the repository root with uv.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic.agent import Agent
from agentic.world_state import WorldState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the agentic sandbox demo.")
    parser.add_argument(
        "--planner",
        choices=("rule_based", "llm"),
        default="rule_based",
        help="Select which planner implementation to use.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    world_state = WorldState(
        robot_position="home",
        object_visible=False,
        holding_object=False,
        target_object="cube",
    )
    agent = Agent(world_state, planner_type=args.planner)
    result = agent.run("pickup_object")

    print("Architecture: goal -> planner -> structured tree -> compiler -> py_trees runtime -> execution")
    print(f"Selected planner: {result['planner_type']}")
    print(f"LangGraph mermaid path: {result['graph_mermaid_path']}")
    print(f"LangGraph image path: {result['graph_image_path']}")
    if result["tree_spec"] is not None:
        print("Structured behavior tree:")
        print(result["tree_spec"].model_dump_json(indent=2))
    print(f"Compile success: {result['compiled_tree'] is not None}")
    print(f"Tree image path: {result['tree_image_path']}")
    print(f"Execution result: {result['execution_status']}")
    if result["error_message"]:
        print(f"Error: {result['error_message']}")
        raise SystemExit(1)
    print(f"Final world state: {world_state}")


if __name__ == "__main__":
    main()
