import py_trees

from agentic.bt_runtime.compiler import compile_behavior_tree
from agentic.bt_runtime.executor import execute_tree
from agentic.bt_runtime.visualization import print_ascii_tree
from agentic.bt_spec.tree_structure import BehaviorTreeStructure
from agentic.planning.planner import create_plan
from agentic.world_state import WorldState


class Agent:
    """Small agent that plans and executes a goal."""

    def __init__(self, world_state: WorldState) -> None:
        self.world_state = world_state

    def plan(self, goal: str) -> BehaviorTreeStructure:
        return create_plan(goal)

    def compile(
        self, structure: BehaviorTreeStructure
    ) -> py_trees.trees.BehaviourTree:
        return compile_behavior_tree(structure, self.world_state)

    def run(self, goal: str) -> None:
        structure = self.plan(goal)
        print("Structured behavior tree:")
        print(structure.model_dump_json(indent=2))
        print(f"Planning for goal: {structure.goal}")
        tree = self.compile(structure)
        print("Runtime behavior tree:")
        print_ascii_tree(tree)
        result = execute_tree(tree)

        if result is py_trees.common.Status.SUCCESS:
            print("Goal achieved")
        else:
            print("Goal failed")
