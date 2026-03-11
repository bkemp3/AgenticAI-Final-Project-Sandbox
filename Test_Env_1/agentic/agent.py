import py_trees

from agentic.bt_runtime.compiler import compile_behavior_tree
from agentic.bt_runtime.executor import execute_tree
from agentic.bt_runtime.visualization import export_tree_image, print_ascii_tree
from agentic.bt_spec.tree_structure import BehaviorTreeStructure
from agentic.planning.base import BasePlanner
from agentic.planning.rule_based_planner import RuleBasedPlanner
from agentic.world_state import WorldState


class Agent:
    """Small agent that plans and executes a goal."""

    def __init__(
        self,
        world_state: WorldState,
        planner: BasePlanner | None = None,
    ) -> None:
        self.world_state = world_state
        self.planner = planner or RuleBasedPlanner()

    def plan(self, goal: str) -> BehaviorTreeStructure:
        return self.planner.create_plan(goal)

    def compile(
        self, structure: BehaviorTreeStructure
    ) -> py_trees.trees.BehaviourTree:
        return compile_behavior_tree(structure, self.world_state)

    def visualize(self, tree: py_trees.trees.BehaviourTree, name: str) -> dict[str, str] | None:
        print("Runtime behavior tree:")
        print_ascii_tree(tree)
        return export_tree_image(tree, name=name)

    def run(self, goal: str) -> None:
        structure = self.plan(goal)
        print("Structured behavior tree:")
        print(structure.model_dump_json(indent=2))
        print(f"Planning for goal: {structure.goal}")
        tree = self.compile(structure)
        artifacts = self.visualize(tree, structure.goal)
        if artifacts:
            print(f"Exported tree artifacts: {artifacts}")
        result = execute_tree(tree)

        if result is py_trees.common.Status.SUCCESS:
            print("Goal achieved")
        else:
            print("Goal failed")
