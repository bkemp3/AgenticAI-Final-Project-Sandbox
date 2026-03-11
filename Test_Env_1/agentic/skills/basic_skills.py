from agentic.behavior_tree.status import Status
from agentic.skills.base_skill import Skill
from agentic.world_state import WorldState


class DetectObject(Skill):
    name = "DetectObject"

    def run(self, world_state: WorldState) -> Status:
        print("DetectObject executed")
        world_state.object_visible = True
        return Status.SUCCESS


class PickObject(Skill):
    name = "PickObject"

    def run(self, world_state: WorldState) -> Status:
        print("PickObject executed")
        if world_state.object_visible:
            world_state.holding_object = True
            return Status.SUCCESS
        return Status.FAILURE
