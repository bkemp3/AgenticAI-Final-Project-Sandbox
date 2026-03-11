from pathlib import Path
import sys

# Allow running the demo directly from the repository root with uv.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic.agent import Agent
from agentic.bt_runtime.executor import execute_tree
from agentic.planning.rule_based_planner import RuleBasedPlanner
from agentic.world_state import WorldState


def main() -> None:
    world_state = WorldState(
        robot_position="home",
        object_visible=False,
        holding_object=False,
        target_object="cube",
    )
    agent = Agent(world_state, planner=RuleBasedPlanner())
    structure = agent.plan("pickup_object")

    print("Structured behavior tree:")
    print(structure.model_dump_json(indent=2))
    print("Architecture: goal -> planner -> structured tree -> compiler -> py_trees runtime -> execution")
    print(f"Planning for goal: {structure.goal}")

    runtime_tree = agent.compile(structure)
    artifacts = agent.visualize(runtime_tree, structure.goal)
    if artifacts:
        print(f"Exported tree artifacts: {artifacts}")

    result = execute_tree(runtime_tree)

    if result.name == "SUCCESS":
        print("Goal achieved")
    else:
        print("Goal failed")

    print(f"Final world state: {world_state}")


if __name__ == "__main__":
    main()
