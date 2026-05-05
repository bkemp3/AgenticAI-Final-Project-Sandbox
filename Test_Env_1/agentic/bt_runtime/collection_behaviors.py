from __future__ import annotations

import py_trees

from agentic.simulation import CollectionWorld, CollectionObservation, VisibleObject

BLACKBOARD_WORLD = "collection_world"
BLACKBOARD_OBSERVATION = "collection_observation"
BLACKBOARD_SELECTED_ID = "selected_object_id"
BLACKBOARD_SELECTED_POSITION = "selected_object_position"


class _CollectionBehaviour(py_trees.behaviour.Behaviour):
    """Base class for BT leaves that operate on partial observations while the simulator owns ground truth."""

    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self.blackboard = py_trees.blackboard.Blackboard()

    def _world(self) -> CollectionWorld:
        world = self._get(BLACKBOARD_WORLD)
        if world is None:
            raise RuntimeError("collection_world is not set on the blackboard")
        return world

    def _observation(self) -> CollectionObservation | None:
        return self._get(BLACKBOARD_OBSERVATION)

    def _selected_object_id(self) -> str | None:
        return self._get(BLACKBOARD_SELECTED_ID)

    def _selected_object_position(self) -> tuple[int, int] | None:
        return self._get(BLACKBOARD_SELECTED_POSITION)

    def _get(self, key: str):
        try:
            return self.blackboard.get(key)
        except KeyError:
            return None


class SenseCollectionWorld(_CollectionBehaviour):
    """Sense the world through the simulator API; the BT receives only partial observation state."""

    def __init__(
        self,
        name: str = "SenseCollectionWorld",
        visibility_radius: int | None = None,
        include_object_details: bool = True,
    ) -> None:
        super().__init__(name=name)
        self.visibility_radius = visibility_radius
        self.include_object_details = include_object_details

    def update(self) -> py_trees.common.Status:
        self.blackboard.set(
            BLACKBOARD_OBSERVATION,
            self._world().get_observation(
                visibility_radius=self.visibility_radius,
                include_object_details=self.include_object_details,
            ),
        )
        return py_trees.common.Status.SUCCESS


class CollectionValueGoalMet(_CollectionBehaviour):
    """Check whether deposited value has reached the configured target while the simulator remains authoritative."""

    def __init__(self, target_value: float | None = None, name: str = "CollectionValueGoalMet") -> None:
        super().__init__(name=name)
        self.target_value = target_value

    def update(self) -> py_trees.common.Status:
        world = self._world()
        target_value = self.target_value if self.target_value is not None else world.state.config.target_value
        if world.state.robot.inventory_value >= target_value:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class TimeRemaining(_CollectionBehaviour):
    """Check the remaining timestep budget from the latest observation."""

    def __init__(self, minimum_remaining: int = 1, name: str = "TimeRemaining") -> None:
        super().__init__(name=name)
        self.minimum_remaining = minimum_remaining

    def update(self) -> py_trees.common.Status:
        observation = self._observation()
        if observation is None:
            return py_trees.common.Status.FAILURE
        if observation.time_remaining >= self.minimum_remaining:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class SelectBestMaterial(_CollectionBehaviour):
    """Select among visible objects only; hidden simulator state is never inspected for target choice."""

    def __init__(self, strategy: str = "highest_value", name: str = "SelectBestMaterial") -> None:
        super().__init__(name=name)
        self.strategy = strategy

    def update(self) -> py_trees.common.Status:
        world = self._world()
        observation = self._observation()
        if observation is None:
            return py_trees.common.Status.FAILURE

        candidates = [visible for visible in observation.visible_objects if self._fits_capacity(world, visible)]
        if not candidates:
            self.blackboard.set(BLACKBOARD_SELECTED_ID, None)
            self.blackboard.set(BLACKBOARD_SELECTED_POSITION, None)
            return py_trees.common.Status.FAILURE

        selected = max(candidates, key=lambda visible: self._score(observation.robot_position, visible))
        self.blackboard.set(BLACKBOARD_SELECTED_ID, selected.object_id)
        self.blackboard.set(BLACKBOARD_SELECTED_POSITION, selected.position)
        return py_trees.common.Status.SUCCESS

    def _fits_capacity(self, world: CollectionWorld, visible: VisibleObject) -> bool:
        if visible.weight is None:
            return True
        return visible.weight <= world.remaining_carry_capacity()

    def _score(self, robot_position: tuple[int, int], visible: VisibleObject) -> tuple[float, float]:
        distance = abs(robot_position[0] - visible.position[0]) + abs(robot_position[1] - visible.position[1])
        value = visible.value if visible.value is not None else 0.0
        weight = visible.weight if visible.weight not in (None, 0) else 1.0
        if self.strategy == "highest_value":
            return value, -distance
        if self.strategy == "value_per_weight":
            return value / weight, value
        if self.strategy == "value_per_distance":
            return value / max(distance, 1), value
        if self.strategy == "nearest":
            return -distance, value
        raise ValueError(f"Unsupported selection strategy: {self.strategy}")


class NavigateToSelectedObject(_CollectionBehaviour):
    """Move one grid step per tick toward the selected visible object through simulator APIs."""

    def __init__(self, name: str = "NavigateToSelectedObject") -> None:
        super().__init__(name=name)

    def update(self) -> py_trees.common.Status:
        target_position = self._selected_object_position()
        if target_position is None:
            return py_trees.common.Status.FAILURE
        world = self._world()
        robot_position = world.state.robot.position
        if robot_position == target_position:
            return py_trees.common.Status.SUCCESS
        next_step = _next_step_toward(robot_position, target_position)
        if not world.move_robot(next_step):
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.RUNNING


class ObjectStillAvailable(_CollectionBehaviour):
    """Confirm the selected object is still present in the latest partial observation."""

    def __init__(self, name: str = "ObjectStillAvailable") -> None:
        super().__init__(name=name)

    def update(self) -> py_trees.common.Status:
        observation = self._observation()
        selected_object_id = self._selected_object_id()
        if observation is None or selected_object_id is None:
            return py_trees.common.Status.FAILURE
        for visible in observation.visible_objects:
            if visible.object_id == selected_object_id:
                self.blackboard.set(BLACKBOARD_SELECTED_POSITION, visible.position)
                return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class AttemptPickupSelectedObject(_CollectionBehaviour):
    """Attempt pickup through the simulator; the BT only sees the success/failure result."""

    def __init__(self, name: str = "AttemptPickupSelectedObject") -> None:
        super().__init__(name=name)

    def update(self) -> py_trees.common.Status:
        selected_object_id = self._selected_object_id()
        if selected_object_id is None:
            return py_trees.common.Status.FAILURE
        if self._world().attempt_pickup(selected_object_id):
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class HoldingObject(_CollectionBehaviour):
    """Check whether the simulator reports the robot is currently holding an object."""

    def __init__(self, name: str = "HoldingObject") -> None:
        super().__init__(name=name)

    def update(self) -> py_trees.common.Status:
        if self._world().state.robot.holding_object_id is not None:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class CarryCapacityAvailable(_CollectionBehaviour):
    """Check whether the robot has enough remaining carry capacity for a visible or selected object."""

    def __init__(self, name: str = "CarryCapacityAvailable") -> None:
        super().__init__(name=name)

    def update(self) -> py_trees.common.Status:
        world = self._world()
        observation = self._observation()
        remaining_capacity = world.remaining_carry_capacity()
        if remaining_capacity <= 0:
            return py_trees.common.Status.FAILURE

        selected_object_id = self._selected_object_id()
        if selected_object_id is not None and observation is not None:
            for visible in observation.visible_objects:
                if visible.object_id == selected_object_id:
                    if visible.weight is None or visible.weight <= remaining_capacity:
                        return py_trees.common.Status.SUCCESS
                    return py_trees.common.Status.FAILURE

        if observation is None:
            return py_trees.common.Status.FAILURE
        if any(visible.weight is None or visible.weight <= remaining_capacity for visible in observation.visible_objects):
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class NavigateToDropoff(_CollectionBehaviour):
    """Move one grid step per tick toward the configured dropoff, leaving state ownership to the simulator."""

    def __init__(self, name: str = "NavigateToDropoff") -> None:
        super().__init__(name=name)

    def update(self) -> py_trees.common.Status:
        world = self._world()
        target_position = world.state.config.dropoff_location
        robot_position = world.state.robot.position
        if robot_position == target_position:
            return py_trees.common.Status.SUCCESS
        next_step = _next_step_toward(robot_position, target_position)
        if not world.move_robot(next_step):
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.RUNNING


class DepositHeldObject(_CollectionBehaviour):
    """Deposit the held object through the simulator API; the BT never mutates inventory state directly."""

    def __init__(self, name: str = "DepositHeldObject") -> None:
        super().__init__(name=name)

    def update(self) -> py_trees.common.Status:
        if self._world().deposit_held_object():
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class ReplanCollectionTarget(_CollectionBehaviour):
    """Clear BT-side target state so the next sense/select pass replans from partial observation."""

    def __init__(self, name: str = "ReplanCollectionTarget") -> None:
        super().__init__(name=name)

    def update(self) -> py_trees.common.Status:
        self.blackboard.set(BLACKBOARD_SELECTED_ID, None)
        self.blackboard.set(BLACKBOARD_SELECTED_POSITION, None)
        return py_trees.common.Status.SUCCESS


def _next_step_toward(start: tuple[int, int], end: tuple[int, int]) -> tuple[int, int]:
    if start[0] != end[0]:
        step_x = start[0] + (1 if end[0] > start[0] else -1)
        return step_x, start[1]
    if start[1] != end[1]:
        step_y = start[1] + (1 if end[1] > start[1] else -1)
        return start[0], step_y
    return start
