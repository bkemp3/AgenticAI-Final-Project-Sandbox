from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

GridPosition = tuple[int, int]


@dataclass(slots=True)
class RobotState:
    position: GridPosition
    holding_object_id: str | None = None
    inventory_weight: float = 0.0
    inventory_value: float = 0.0


@dataclass(slots=True)
class MaterialObject:
    object_id: str
    position: GridPosition
    weight: float
    value: float
    available: bool = True
    pickup_failure_prob: float = 0.0
    collected_by: str | None = None


@dataclass(slots=True)
class CollectionConfig:
    grid_size: GridPosition
    target_weight: float
    max_timesteps: int
    dropoff_location: GridPosition
    disappear_prob: float
    seed: int | None = None


@dataclass(slots=True)
class CollectionWorldState:
    timestep: int
    config: CollectionConfig
    robot: RobotState
    objects: dict[str, MaterialObject] = field(default_factory=dict)


@dataclass(slots=True)
class WorldEvent:
    timestep: int
    event_type: str
    object_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VisibleObject:
    object_id: str
    position: GridPosition
    weight: float | None = None
    value: float | None = None


@dataclass(frozen=True, slots=True)
class CollectionObservation:
    robot_position: GridPosition
    held_object_id: str | None
    collected_weight: float
    collected_value: float
    time_remaining: int
    visibility_radius: int | None
    visible_objects: tuple[VisibleObject, ...]


@dataclass(frozen=True, slots=True)
class CollectionMetrics:
    success: bool
    total_weight_collected: float
    total_value_collected: float
    timesteps_used: int
    failed_pickups: int
    successful_pickups: int
    deposits: int
    objects_lost_to_other_agents: int
    invalid_actions: int
    final_inventory_weight: float
    final_inventory_value: float
