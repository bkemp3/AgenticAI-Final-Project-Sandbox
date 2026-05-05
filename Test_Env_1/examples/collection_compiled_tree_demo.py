import sys
from pathlib import Path

import py_trees

# Allow running the demo directly from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic.behaviors import load_behavior_catalog
from agentic.bt_runtime.compiler import compile_behavior_tree
from agentic.bt_spec.nodes import LeafNode, ParamEntry, SelectorNode, SequenceNode
from agentic.bt_spec.tree_structure import BehaviorTreeStructure
from agentic.simulation import CollectionWorld, load_collection_config


def build_generated_style_tree() -> BehaviorTreeStructure:
    return BehaviorTreeStructure(
        goal="maximize_collection_value",
        description="Generated-style collection tree that chooses value-efficient objects and deposits them.",
        root=SelectorNode(
            type="selector",
            name="goal_or_work",
            children=[
                LeafNode(type="collection_value_goal_met"),
                SequenceNode(
                    type="sequence",
                    name="work_loop",
                    children=[
                        LeafNode(
                            type="sense_collection_world",
                            params=[
                                ParamEntry(key="visibility_radius", value=None),
                                ParamEntry(key="include_object_details", value=True),
                            ],
                        ),
                        LeafNode(
                            type="time_remaining",
                            params=[ParamEntry(key="minimum_remaining", value=1)],
                        ),
                        SelectorNode(
                            type="selector",
                            name="deposit_or_collect",
                            children=[
                                SequenceNode(
                                    type="sequence",
                                    name="deposit_sequence",
                                    children=[
                                        LeafNode(type="holding_object"),
                                        LeafNode(type="navigate_to_dropoff"),
                                        LeafNode(type="deposit_held_object"),
                                        LeafNode(type="replan_collection_target"),
                                    ],
                                ),
                                SequenceNode(
                                    type="sequence",
                                    name="collect_sequence",
                                    children=[
                                        LeafNode(type="carry_capacity_available"),
                                        LeafNode(
                                            type="select_best_material",
                                            params=[ParamEntry(key="strategy", value="value_per_distance")],
                                        ),
                                        LeafNode(type="navigate_to_selected_object"),
                                        LeafNode(
                                            type="sense_collection_world",
                                            params=[
                                                ParamEntry(key="visibility_radius", value=None),
                                                ParamEntry(key="include_object_details", value=True),
                                            ],
                                        ),
                                        LeafNode(type="object_still_available"),
                                        LeafNode(type="attempt_pickup_selected_object"),
                                    ],
                                ),
                                LeafNode(type="replan_collection_target"),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    behavior_catalog = load_behavior_catalog(str(root / "behavior_sets" / "collection.yaml"))
    config, objects = load_collection_config(str(root / "configs" / "collection_env.yaml"))
    world = CollectionWorld(config=config, objects=objects)
    tree_spec = build_generated_style_tree()
    tree = compile_behavior_tree(tree_spec, world, behavior_catalog.runtime_registry())

    print(f"Behavior set: {behavior_catalog.name}")
    print(f"Environment config: {config}")
    printed_events = 0
    for tick in range(config.max_timesteps):
        if world.is_terminal():
            break
        tree.tick()
        blackboard = py_trees.blackboard.Blackboard()
        print(
            f"tick={tick} status={tree.root.status} selected_object={blackboard.get('/selected_object_id')} "
            f"position={world.state.robot.position} held={world.state.robot.holding_object_id}"
        )
        for event in world.events[printed_events:]:
            print(f"  event: {event}")
        printed_events = len(world.events)

    print(f"Final metrics: {world.get_metrics()}")


if __name__ == "__main__":
    main()
