from agentic.bt_spec.tree_structure import BehaviorTreeStructure
from agentic.planning.planner import compile_structure, create_plan
from agentic.runtime.executor import execute_tree
from agentic.world_state import WorldState


class Agent:
    """Small agent that plans and executes a goal."""

    def __init__(self, world_state: WorldState) -> None:
        self.world_state = world_state

    def plan(self, goal: str) -> BehaviorTreeStructure:
        return create_plan(goal)

    def run(self, goal: str) -> None:
        structure = self.plan(goal)
        self.run_structure(structure)

    def run_structure(self, structure: BehaviorTreeStructure) -> None:
        print(f"Planning for goal: {structure.goal}")
        tree = compile_structure(structure)
        result = execute_tree(tree, self.world_state)

        if result.name == "SUCCESS":
            print("Goal achieved")
        else:
            print("Goal failed")
