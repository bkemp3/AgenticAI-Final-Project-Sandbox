from __future__ import annotations

from dataclasses import dataclass

import py_trees

from agentic.bt_runtime.collection_behaviors import BLACKBOARD_SELECTED_ID
from agentic.bt_spec.nodes import LeafNode, SequenceNode
from agentic.bt_spec.tree_structure import BehaviorTreeStructure
from agentic.planning.prompting import summarize_collection_config
from agentic.simulation import CollectionWorld, load_collection_config


@dataclass(frozen=True)
class CollectionTaskAdapter:
    def load_world(self, environment_config_path: str) -> CollectionWorld:
        config, objects = load_collection_config(environment_config_path)
        return CollectionWorld(config=config, objects=objects)

    def summarize_world(self, world: CollectionWorld) -> str:
        return summarize_collection_config(world.state.config)

    def describe_tick(self, world: CollectionWorld, tree: py_trees.trees.BehaviourTree, tick: int) -> str:
        blackboard = py_trees.blackboard.Blackboard()
        selected_object = self._safe_blackboard_get(blackboard, f"/{BLACKBOARD_SELECTED_ID}")
        return (
            f"tick={tick} status={tree.root.status} selected_object={selected_object} "
            f"position={world.state.robot.position} held={world.state.robot.holding_object_id}"
        )

    def get_events(self, world: CollectionWorld) -> list[object]:
        return list(world.events)

    def get_metrics(self, world: CollectionWorld) -> object:
        return world.get_metrics()

    def is_terminal(self, world: CollectionWorld) -> bool:
        return world.is_terminal()

    def validate_plan(self, tree_spec: BehaviorTreeStructure) -> str | None:
        root = tree_spec.root
        if isinstance(root, SequenceNode):
            for child in root.children:
                if isinstance(child, LeafNode) and child.type == "collection_value_goal_met":
                    return (
                        "collection_value_goal_met cannot be a direct child of the top-level sequence because it "
                        "fails until the goal is already satisfied and prevents all collection work from running. "
                        "Use a top-level selector with the goal check as one branch and the work loop as another."
                    )
                if not isinstance(child, LeafNode):
                    break
        return None

    @staticmethod
    def _safe_blackboard_get(blackboard: py_trees.blackboard.Blackboard, key: str):
        try:
            return blackboard.get(key)
        except KeyError:
            return None


adapter = CollectionTaskAdapter()
