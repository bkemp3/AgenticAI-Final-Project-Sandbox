import sys
from pathlib import Path

import py_trees

# Allow running the demo directly from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic.bt_runtime.collection_behaviors import (
    BLACKBOARD_SELECTED_ID,
    BLACKBOARD_WORLD,
    AttemptPickupSelectedObject,
    CarryCapacityAvailable,
    CollectionValueGoalMet,
    DepositHeldObject,
    HoldingObject,
    NavigateToDropoff,
    NavigateToSelectedObject,
    ObjectStillAvailable,
    ReplanCollectionTarget,
    SelectBestMaterial,
    SenseCollectionWorld,
    TimeRemaining,
)
from agentic.simulation import CollectionWorld, load_collection_config


def build_tree() -> py_trees.trees.BehaviourTree:
    goal_or_work = py_trees.composites.Selector(
        name="GoalOrWork",
        memory=False,
        children=[
            CollectionValueGoalMet(),
            py_trees.composites.Sequence(
                name="WorkLoop",
                memory=False,
                children=[
                    SenseCollectionWorld(visibility_radius=None, include_object_details=True),
                    TimeRemaining(minimum_remaining=1),
                    py_trees.composites.Selector(
                        name="DepositOrCollect",
                        memory=False,
                        children=[
                            py_trees.composites.Sequence(
                                name="DepositSequence",
                                memory=False,
                                children=[
                                    HoldingObject(),
                                    NavigateToDropoff(),
                                    DepositHeldObject(),
                                    ReplanCollectionTarget(),
                                ],
                            ),
                            py_trees.composites.Sequence(
                                name="CollectSequence",
                                memory=False,
                                children=[
                                    CarryCapacityAvailable(),
                                    SelectBestMaterial(strategy="value_per_distance"),
                                    NavigateToSelectedObject(),
                                    SenseCollectionWorld(visibility_radius=None, include_object_details=True),
                                    ObjectStillAvailable(),
                                    AttemptPickupSelectedObject(),
                                ],
                            ),
                            ReplanCollectionTarget(),
                        ],
                    ),
                ],
            ),
        ],
    )
    return py_trees.trees.BehaviourTree(root=goal_or_work)


def main() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "collection_env.yaml"
    config, objects = load_collection_config(str(config_path))
    world = CollectionWorld(config=config, objects=objects)

    blackboard = py_trees.blackboard.Blackboard()
    blackboard.set(BLACKBOARD_WORLD, world)
    blackboard.set(BLACKBOARD_SELECTED_ID, None)

    tree = build_tree()
    max_ticks = config.max_timesteps
    printed_events = 0

    print(f"Loaded config: {config}")
    for tick in range(max_ticks):
        if world.is_terminal():
            break
        tree.tick()
        selected_object = blackboard.get(BLACKBOARD_SELECTED_ID)
        print(
            f"tick={tick} status={tree.root.status} selected_object={selected_object} "
            f"robot_position={world.state.robot.position} held={world.state.robot.holding_object_id}"
        )
        for event in world.events[printed_events:]:
            print(f"  event: {event}")
        printed_events = len(world.events)

    print(f"Final metrics: {world.get_metrics()}")


if __name__ == "__main__":
    main()
