from agentic.bt_spec.nodes import DetectObjectNode, PickObjectNode, SequenceNode
from agentic.bt_spec.tree_structure import BehaviorTreeStructure
from agentic.planning.base import BasePlanner


class RuleBasedPlanner(BasePlanner):
    """Baseline planner with a small hand-authored goal mapping."""

    def create_plan(self, goal: str) -> BehaviorTreeStructure:
        if goal == "pickup_object":
            return BehaviorTreeStructure(
                goal=goal,
                description="Detect the target object, then pick it up.",
                root=SequenceNode(
                    type="sequence",
                    name="pickup_sequence",
                    children=[
                        DetectObjectNode(type="detect_object"),
                        PickObjectNode(type="pick_object"),
                    ],
                ),
            )

        raise ValueError(f"Unknown goal: {goal}")
