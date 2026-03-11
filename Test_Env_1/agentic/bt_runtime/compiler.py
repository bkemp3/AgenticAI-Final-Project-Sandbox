import py_trees

from agentic.bt_runtime.registry import LEAF_BEHAVIOUR_REGISTRY
from agentic.bt_spec.nodes import (
    BehaviorTreeNode,
    DetectObjectNode,
    HoldingObjectNode,
    ObjectVisibleNode,
    PickObjectNode,
    SelectorNode,
    SequenceNode,
)
from agentic.bt_spec.tree_structure import BehaviorTreeStructure
from agentic.world_state import WorldState


def compile_behavior_tree(
    tree_spec: BehaviorTreeStructure, world_state: WorldState
) -> py_trees.trees.BehaviourTree:
    """Compile a validated tree spec into a py_trees behavior tree."""

    root = _compile_node(tree_spec.root, world_state)
    return py_trees.trees.BehaviourTree(root=root)


def _compile_node(
    node: BehaviorTreeNode, world_state: WorldState
) -> py_trees.behaviour.Behaviour:
    if isinstance(node, SequenceNode):
        return py_trees.composites.Sequence(
            name=node.name or "Sequence",
            memory=False,
            children=[_compile_node(child, world_state) for child in node.children],
        )
    if isinstance(node, SelectorNode):
        return py_trees.composites.Selector(
            name=node.name or "Selector",
            memory=False,
            children=[_compile_node(child, world_state) for child in node.children],
        )
    if isinstance(
        node,
        (DetectObjectNode, PickObjectNode, ObjectVisibleNode, HoldingObjectNode),
    ):
        behaviour_class = LEAF_BEHAVIOUR_REGISTRY.get(node.type)
        if behaviour_class is None:
            raise ValueError(f"No runtime behaviour registered for node type: {node.type}")
        kwargs = {"world_state": world_state, "name": node.name or _default_name(node.type)}
        if isinstance(node, PickObjectNode):
            kwargs["target"] = node.target
        return behaviour_class(**kwargs)

    raise ValueError(f"Unsupported schema node type: {node.type}")


def _default_name(node_type: str) -> str:
    return "".join(part.capitalize() for part in node_type.split("_"))
