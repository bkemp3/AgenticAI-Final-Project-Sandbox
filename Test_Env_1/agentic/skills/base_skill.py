from agentic.behavior_tree.status import Status
from agentic.world_state import WorldState


class Skill:
    """Simple skill interface."""

    name: str

    def run(self, world_state: WorldState) -> Status:
        raise NotImplementedError
