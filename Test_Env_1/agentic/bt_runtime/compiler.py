import inspect

import py_trees

from agentic.bt_spec.nodes import (
    BehaviorTreeNode,
    LeafNode,
    SelectorNode,
    SequenceNode,
)
from agentic.bt_spec.tree_structure import BehaviorTreeStructure
from agentic.bt_runtime.collection_behaviors import BLACKBOARD_WORLD
from agentic.world_state import WorldState


def compile_behavior_tree(
    tree_spec: BehaviorTreeStructure,
    world_state: object,
    leaf_behaviour_registry: dict[str, type[py_trees.behaviour.Behaviour]],
) -> py_trees.trees.BehaviourTree:
    """Compile a validated tree spec into a py_trees behavior tree."""

    py_trees.blackboard.Blackboard.clear()
    root = _compile_node(tree_spec.root, world_state, leaf_behaviour_registry)
    tree = py_trees.trees.BehaviourTree(root=root)
    py_trees.blackboard.Blackboard().set(BLACKBOARD_WORLD, world_state)
    return tree


def _compile_node(
    node: BehaviorTreeNode,
    world_state: object,
    leaf_behaviour_registry: dict[str, type[py_trees.behaviour.Behaviour]],
) -> py_trees.behaviour.Behaviour:
    if isinstance(node, SequenceNode):
        return py_trees.composites.Sequence(
            name=node.name or "Sequence",
            memory=False,
            children=[
                _compile_node(child, world_state, leaf_behaviour_registry)
                for child in node.children
            ],
        )
    if isinstance(node, SelectorNode):
        return py_trees.composites.Selector(
            name=node.name or "Selector",
            memory=False,
            children=[
                _compile_node(child, world_state, leaf_behaviour_registry)
                for child in node.children
            ],
        )
    if isinstance(node, LeafNode):
        behaviour_class = leaf_behaviour_registry.get(node.type)
        if behaviour_class is None:
            raise ValueError(f"No runtime behaviour registered for node type: {node.type}")
        kwargs = {"name": node.name or _default_name(node.type)}
        if _accepts_world_state(behaviour_class):
            kwargs["world_state"] = world_state
        kwargs.update({entry.key: entry.value for entry in node.params})
        return behaviour_class(**kwargs)

    raise ValueError(f"Unsupported schema node type: {node.type}")


def _default_name(node_type: str) -> str:
    return "".join(part.capitalize() for part in node_type.split("_"))


def _accepts_world_state(behaviour_class: type[py_trees.behaviour.Behaviour]) -> bool:
    signature = inspect.signature(behaviour_class.__init__)
    return "world_state" in signature.parameters
