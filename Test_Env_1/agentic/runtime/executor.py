from agentic.behavior_tree.status import Status
from agentic.behavior_tree.tree import BehaviorTree
from agentic.world_state import WorldState


def execute_tree(tree: BehaviorTree, world_state: WorldState) -> Status:
    """Tick until the tree reaches a terminal state."""

    while True:
        result = tree.tick(world_state)
        if result in (Status.SUCCESS, Status.FAILURE):
            return result
