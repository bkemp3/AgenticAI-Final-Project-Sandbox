from __future__ import annotations

from abc import ABC, abstractmethod

from agentic.bt_spec.tree_structure import BehaviorTreeStructure
from agentic.runtime_feedback import PlannerRepairRequest


class PlannerError(RuntimeError):
    """Raised when a planner cannot produce a valid behavior tree structure."""


class BasePlanner(ABC):
    """Planner interface for producing validated behavior tree structures."""

    @abstractmethod
    def create_plan(self, goal: str) -> BehaviorTreeStructure:
        """Build a validated behavior tree structure for a goal."""

    def repair_plan(
        self,
        goal: str,
        current_tree: BehaviorTreeStructure,
        repair_request: PlannerRepairRequest,
    ) -> BehaviorTreeStructure:
        raise PlannerError(f"{self.__class__.__name__} does not support runtime tree repair.")
