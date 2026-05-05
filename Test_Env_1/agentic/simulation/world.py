from __future__ import annotations

import copy
import random

from agentic.simulation.models import (
    CollectionConfig,
    CollectionMetrics,
    CollectionObservation,
    CollectionWorldState,
    GridPosition,
    MaterialObject,
    RobotState,
    VisibleObject,
    WorldEvent,
)


class CollectionWorld:
    """Owns hidden simulator state for timed collection tasks."""

    def __init__(
        self,
        config: CollectionConfig,
        objects: list[MaterialObject],
        robot_start: GridPosition | None = None,
    ) -> None:
        self._validate_config(config)
        self._rng = random.Random(config.seed)
        self._failed_pickups = 0
        self._successful_pickups = 0
        self._deposits = 0
        self._objects_lost_to_other_agents = 0
        self._invalid_actions = 0
        robot_position = robot_start if robot_start is not None else config.dropoff_location
        self._validate_position(robot_position, config.grid_size, field_name="robot_start")

        world_objects = {obj.object_id: copy.deepcopy(obj) for obj in objects}
        if len(world_objects) != len(objects):
            raise ValueError("Object IDs must be unique.")
        for obj in world_objects.values():
            self._validate_position(obj.position, config.grid_size, field_name=f"object {obj.object_id}")

        self.state = CollectionWorldState(
            timestep=0,
            config=config,
            robot=RobotState(position=robot_position),
            objects=world_objects,
        )
        self.events: list[WorldEvent] = []

    def step(self) -> None:
        if self.is_terminal():
            self._log_invalid_action("step", reason="world is already terminal")
            return
        self.events.append(
            WorldEvent(
                timestep=self.state.timestep,
                event_type="step",
                details={
                    "robot_position": self.state.robot.position,
                    "held_object_id": self.state.robot.holding_object_id,
                },
            )
        )
        self._advance_time()

    def simulate_other_agents(self) -> int:
        """Public hook for applying disappearance dynamics outside an action."""
        return self._apply_disappearances()

    def move_robot(self, new_position: GridPosition) -> bool:
        if self.is_terminal():
            self._log_invalid_action("move_robot", reason="world is already terminal")
            return False
        if not self._is_in_bounds(new_position):
            self._log_invalid_action("move_robot", reason="target position out of bounds", position=new_position)
            self._advance_time()
            return False
        if self._manhattan_distance(self.state.robot.position, new_position) != 1:
            self._log_invalid_action(
                "move_robot",
                reason="robot can only move one grid cell per timestep",
                position=new_position,
            )
            self._advance_time()
            return False

        previous_position = self.state.robot.position
        self.state.robot.position = new_position
        self.events.append(
            WorldEvent(
                timestep=self.state.timestep,
                event_type="move",
                details={
                    "from": previous_position,
                    "to": new_position,
                    "held_object_id": self.state.robot.holding_object_id,
                },
            )
        )
        self._advance_time()
        return True

    def attempt_pickup(self, object_id: str) -> bool:
        if self.is_terminal():
            self._log_invalid_action("attempt_pickup", object_id=object_id, reason="world is already terminal")
            return False
        if self.state.robot.holding_object_id is not None:
            self._log_invalid_action(
                "attempt_pickup",
                object_id=object_id,
                reason="robot is already holding an object",
            )
            self._advance_time()
            return False

        target = self.state.objects.get(object_id)
        if target is None:
            self._log_invalid_action("attempt_pickup", object_id=object_id, reason="unknown object")
            self._advance_time()
            return False
        if not target.available:
            self._log_invalid_action(
                "attempt_pickup",
                object_id=object_id,
                reason="object is not available",
            )
            self._advance_time()
            return False
        if target.position != self.state.robot.position:
            self._log_invalid_action(
                "attempt_pickup",
                object_id=object_id,
                reason="robot is not at the object's position",
            )
            self._advance_time()
            return False

        if self._rng.random() < target.pickup_failure_prob:
            self._failed_pickups += 1
            self.events.append(
                WorldEvent(
                    timestep=self.state.timestep,
                    event_type="pickup_failed",
                    object_id=object_id,
                    details={
                        "pickup_failure_prob": target.pickup_failure_prob,
                        "robot_position": self.state.robot.position,
                        "object_position": target.position,
                    },
                )
            )
            self._advance_time()
            return False

        target.available = False
        target.collected_by = "robot"
        self.state.robot.holding_object_id = object_id
        self._successful_pickups += 1
        self.events.append(
            WorldEvent(
                timestep=self.state.timestep,
                event_type="pickup_success",
                object_id=object_id,
                details={
                    "weight": target.weight,
                    "value": target.value,
                    "robot_position": self.state.robot.position,
                    "object_position": target.position,
                },
            )
        )
        self._advance_time()
        return True

    def deposit_object(self) -> bool:
        if self.is_terminal():
            self._log_invalid_action("deposit_object", reason="world is already terminal")
            return False
        held_object_id = self.state.robot.holding_object_id
        if held_object_id is None:
            self._log_invalid_action("deposit_object", reason="robot is not holding an object")
            self._advance_time()
            return False
        if self.state.robot.position != self.state.config.dropoff_location:
            self._log_invalid_action(
                "deposit_object",
                object_id=held_object_id,
                reason="robot is not at the dropoff location",
            )
            self._advance_time()
            return False

        held_object = self.state.objects[held_object_id]
        held_object.position = self.state.config.dropoff_location
        self.state.robot.inventory_weight += held_object.weight
        self.state.robot.inventory_value += held_object.value
        self.state.robot.holding_object_id = None
        self._deposits += 1
        self.events.append(
            WorldEvent(
                timestep=self.state.timestep,
                event_type="deposit",
                object_id=held_object_id,
                details={
                    "weight": held_object.weight,
                    "value": held_object.value,
                    "dropoff_location": self.state.config.dropoff_location,
                    "inventory_weight": self.state.robot.inventory_weight,
                    "inventory_value": self.state.robot.inventory_value,
                },
            )
        )
        self._advance_time()
        return True

    def goal_met(self) -> bool:
        return self.state.robot.inventory_weight >= self.state.config.target_weight

    def is_terminal(self) -> bool:
        return self.goal_met() or self.state.timestep >= self.state.config.max_timesteps

    def get_observation(
        self,
        visibility_radius: int | None = None,
        include_object_details: bool = True,
    ) -> CollectionObservation:
        if visibility_radius is not None and visibility_radius < 0:
            raise ValueError("visibility_radius cannot be negative.")
        visible_objects = tuple(
            VisibleObject(
                object_id=obj.object_id,
                position=obj.position,
                weight=obj.weight if include_object_details else None,
                value=obj.value if include_object_details else None,
            )
            for obj in sorted(self.state.objects.values(), key=lambda item: item.object_id)
            if obj.available and self._is_visible(obj.position, visibility_radius)
        )
        return CollectionObservation(
            robot_position=self.state.robot.position,
            held_object_id=self.state.robot.holding_object_id,
            collected_weight=self.state.robot.inventory_weight,
            collected_value=self.state.robot.inventory_value,
            time_remaining=max(self.state.config.max_timesteps - self.state.timestep, 0),
            visibility_radius=visibility_radius,
            visible_objects=visible_objects,
        )

    def get_metrics(self) -> CollectionMetrics:
        return CollectionMetrics(
            success=self.goal_met() and self.state.timestep <= self.state.config.max_timesteps,
            total_weight_collected=self.state.robot.inventory_weight,
            total_value_collected=self.state.robot.inventory_value,
            timesteps_used=self.state.timestep,
            failed_pickups=self._failed_pickups,
            successful_pickups=self._successful_pickups,
            deposits=self._deposits,
            objects_lost_to_other_agents=self._objects_lost_to_other_agents,
            invalid_actions=self._invalid_actions,
            final_inventory_weight=self.state.robot.inventory_weight,
            final_inventory_value=self.state.robot.inventory_value,
        )

    def _advance_time(self) -> None:
        self.state.timestep += 1
        self._apply_disappearances()

    def _apply_disappearances(self) -> int:
        disappeared_count = 0
        for obj in self.state.objects.values():
            if not obj.available:
                continue
            if self._rng.random() >= self.state.config.disappear_prob:
                continue
            obj.available = False
            obj.collected_by = "other_agent"
            self._objects_lost_to_other_agents += 1
            disappeared_count += 1
            self.events.append(
                WorldEvent(
                    timestep=self.state.timestep,
                    event_type="object_disappeared",
                    object_id=obj.object_id,
                    details={
                        "position": obj.position,
                        "disappear_prob": self.state.config.disappear_prob,
                    },
                )
            )
        return disappeared_count

    def _log_invalid_action(
        self,
        action: str,
        object_id: str | None = None,
        **details: object,
    ) -> None:
        self._invalid_actions += 1
        self.events.append(
            WorldEvent(
                timestep=self.state.timestep,
                event_type="invalid_action",
                object_id=object_id,
                details={
                    "action": action,
                    "robot_position": self.state.robot.position,
                    "held_object_id": self.state.robot.holding_object_id,
                    **details,
                },
            )
        )

    def _is_in_bounds(self, position: GridPosition) -> bool:
        width, height = self.state.config.grid_size
        x, y = position
        return 0 <= x < width and 0 <= y < height

    def _is_visible(self, position: GridPosition, visibility_radius: int | None) -> bool:
        if visibility_radius is None:
            return True
        return self._manhattan_distance(self.state.robot.position, position) <= visibility_radius

    @staticmethod
    def _manhattan_distance(start: GridPosition, end: GridPosition) -> int:
        return abs(start[0] - end[0]) + abs(start[1] - end[1])

    @staticmethod
    def _validate_config(config: CollectionConfig) -> None:
        if config.max_timesteps <= 0:
            raise ValueError("max_timesteps must be positive.")
        if config.target_weight < 0:
            raise ValueError("target_weight cannot be negative.")
        CollectionWorld._validate_position(
            config.dropoff_location,
            config.grid_size,
            field_name="dropoff_location",
        )

    @staticmethod
    def _validate_position(
        position: GridPosition,
        grid_size: GridPosition,
        field_name: str,
    ) -> None:
        width, height = grid_size
        x, y = position
        if width <= 0 or height <= 0:
            raise ValueError("grid_size dimensions must be positive.")
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"{field_name} must be inside the grid bounds {grid_size}.")
