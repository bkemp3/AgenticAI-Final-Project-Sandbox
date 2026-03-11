import py_trees

from agentic.world_state import WorldState


class DetectObjectBehaviour(py_trees.behaviour.Behaviour):
    """Action that marks the target object as visible."""

    def __init__(self, world_state: WorldState, name: str = "DetectObject") -> None:
        super().__init__(name=name)
        self.world_state = world_state

    def update(self) -> py_trees.common.Status:
        print("DetectObject executed")
        self.world_state.object_visible = True
        return py_trees.common.Status.SUCCESS


class PickObjectBehaviour(py_trees.behaviour.Behaviour):
    """Action that picks up the visible object."""

    def __init__(
        self,
        world_state: WorldState,
        name: str = "PickObject",
        target: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.world_state = world_state
        self.target = target

    def update(self) -> py_trees.common.Status:
        print("PickObject executed")
        if self.world_state.object_visible:
            self.world_state.holding_object = True
            if self.target is not None:
                self.world_state.target_object = self.target
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class ObjectVisibleBehaviour(py_trees.behaviour.Behaviour):
    """Condition that checks whether an object is visible."""

    def __init__(self, world_state: WorldState, name: str = "ObjectVisible") -> None:
        super().__init__(name=name)
        self.world_state = world_state

    def update(self) -> py_trees.common.Status:
        print("ObjectVisible check")
        if self.world_state.object_visible:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class HoldingObjectBehaviour(py_trees.behaviour.Behaviour):
    """Condition that checks whether the robot is holding an object."""

    def __init__(self, world_state: WorldState, name: str = "HoldingObject") -> None:
        super().__init__(name=name)
        self.world_state = world_state

    def update(self) -> py_trees.common.Status:
        print("HoldingObject check")
        if self.world_state.holding_object:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE
