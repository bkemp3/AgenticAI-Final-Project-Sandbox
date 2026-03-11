from agentic.behavior_tree.nodes import ActionNode, Node, Sequence
from agentic.behavior_tree.tree import BehaviorTree
from agentic.bt_spec.nodes import (
    BehaviorTreeNode,
    DetectObjectNode,
    PickObjectNode,
    SequenceNode,
)
from agentic.bt_spec.tree_structure import BehaviorTreeStructure
from agentic.skills.basic_skills import DetectObject, PickObject


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


def compile_structure(structure: BehaviorTreeStructure) -> BehaviorTree:
    """Compile a validated spec into the current minimal runtime tree."""

    return BehaviorTree(_compile_node(structure.root))


def _compile_node(node: BehaviorTreeNode) -> Node:
    if isinstance(node, SequenceNode):
        return Sequence([_compile_node(child) for child in node.children])
    if isinstance(node, DetectObjectNode):
        return ActionNode(DetectObject())
    if isinstance(node, PickObjectNode):
        return ActionNode(PickObject())

    raise ValueError(f"Unsupported runtime node type: {node.type}")
