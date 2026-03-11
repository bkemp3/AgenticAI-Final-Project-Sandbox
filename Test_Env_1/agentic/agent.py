from agentic.planning.planner import create_plan
from agentic.runtime.executor import execute_tree
from agentic.world_state import WorldState


class Agent:
    """Small agent that plans and executes a goal."""

    def __init__(self, world_state: WorldState) -> None:
        self.world_state = world_state

    def run(self, goal: str) -> None:
        print(f"Planning for goal: {goal}")
        tree = create_plan(goal)
        result = execute_tree(tree, self.world_state)

        if result.name == "SUCCESS":
            print("Goal achieved")
        else:
            print("Goal failed")
