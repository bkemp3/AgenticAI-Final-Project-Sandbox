from agentic.bt_spec.nodes import DetectObjectNode, PickObjectNode, SequenceNode
from agentic.bt_spec.tree_structure import BehaviorTreeStructure


def create_plan(goal: str) -> BehaviorTreeStructure:
    """Create a validated behavior tree structure for a known goal."""

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
