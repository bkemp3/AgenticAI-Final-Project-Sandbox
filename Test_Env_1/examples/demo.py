from pathlib import Path
import sys

# Allow running the demo directly from the repository root with uv.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic.agent import Agent
from agentic.world_state import WorldState


def main() -> None:
    world_state = WorldState(
        robot_position="home",
        object_visible=False,
        holding_object=False,
        target_object="cube",
    )
    agent = Agent(world_state)
    structure = agent.plan("pickup_object")

    print("Structured behavior tree:")
    print(structure.model_dump_json(indent=2))
    print("Architecture: goal -> planner -> structured tree -> compiler -> py_trees runtime -> execution")
    print(f"Planning for goal: {structure.goal}")

    runtime_tree = agent.compile(structure)
    print("Runtime behavior tree:")
    from agentic.bt_runtime.visualization import print_ascii_tree

    print_ascii_tree(runtime_tree)

    from agentic.bt_runtime.executor import execute_tree

    result = execute_tree(runtime_tree)

    if result.name == "SUCCESS":
        print("Goal achieved")
    else:
        print("Goal failed")

    print(f"Final world state: {world_state}")


if __name__ == "__main__":
    main()
