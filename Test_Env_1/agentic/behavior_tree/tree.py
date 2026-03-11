from agentic.behavior_tree.nodes import Node
from agentic.behavior_tree.status import Status
from agentic.world_state import WorldState


class BehaviorTree:
    """Minimal behavior tree wrapper around a root node."""

    def __init__(self, root: Node) -> None:
        self.root = root

    def tick(self, world_state: WorldState) -> Status:
        return self.root.tick(world_state)
