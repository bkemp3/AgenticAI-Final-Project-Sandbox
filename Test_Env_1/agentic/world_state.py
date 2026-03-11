from dataclasses import dataclass


@dataclass
class WorldState:
    """Simple environment state shared across the agent."""

    robot_position: str
    object_visible: bool
    holding_object: bool
    target_object: str | None
