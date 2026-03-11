from agentic.behavior_tree.nodes import ActionNode, Sequence
from agentic.behavior_tree.tree import BehaviorTree
from agentic.skills.basic_skills import DetectObject, PickObject


def create_plan(goal: str) -> BehaviorTree:
    """Create a behavior tree for a known goal."""

    if goal == "pickup_object":
        root = Sequence(
            [
                ActionNode(DetectObject()),
                ActionNode(PickObject()),
            ]
        )
        return BehaviorTree(root)

    raise ValueError(f"Unknown goal: {goal}")
