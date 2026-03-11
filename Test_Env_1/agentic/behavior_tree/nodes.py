from __future__ import annotations

from collections.abc import Sequence as SequenceCollection

from agentic.behavior_tree.status import Status
from agentic.skills.base_skill import Skill
from agentic.world_state import WorldState


class Node:
    """Base node for behavior tree execution."""

    def tick(self, world_state: WorldState) -> Status:
        raise NotImplementedError


class Sequence(Node):
    """Run children in order until one fails or is still running."""

    def __init__(self, children: SequenceCollection[Node]) -> None:
        self.children = list(children)

    def tick(self, world_state: WorldState) -> Status:
        for child in self.children:
            status = child.tick(world_state)
            if status is Status.FAILURE:
                return Status.FAILURE
            if status is Status.RUNNING:
                return Status.RUNNING
        return Status.SUCCESS


class ActionNode(Node):
    """Wrap a skill as a behavior tree action."""

    def __init__(self, skill: Skill) -> None:
        self.skill = skill

    def tick(self, world_state: WorldState) -> Status:
        return self.skill.run(world_state)
