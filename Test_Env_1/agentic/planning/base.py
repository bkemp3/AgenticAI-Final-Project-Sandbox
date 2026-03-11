from __future__ import annotations

from abc import ABC, abstractmethod

from agentic.bt_spec.tree_structure import BehaviorTreeStructure


class BasePlanner(ABC):
    """Planner interface for producing validated behavior tree structures."""

    @abstractmethod
    def create_plan(self, goal: str) -> BehaviorTreeStructure:
        """Build a validated behavior tree structure for a goal."""
