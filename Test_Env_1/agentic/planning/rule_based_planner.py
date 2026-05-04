from agentic.behaviors.catalog import BehaviorCatalog
from agentic.bt_spec.nodes import LeafNode, SequenceNode
from agentic.bt_spec.tree_structure import BehaviorTreeStructure
from agentic.planning.base import BasePlanner, PlannerError


class RuleBasedPlanner(BasePlanner):
    """Baseline planner with a small hand-authored goal mapping."""

    def __init__(self, behavior_catalog: BehaviorCatalog) -> None:
        self.behavior_catalog = behavior_catalog

    def create_plan(self, goal: str) -> BehaviorTreeStructure:
        available_types = set(self.behavior_catalog.leaf_types)
        if goal == "pickup_object":
            required_types = {"detect_object", "pick_object"}
            missing_types = required_types - available_types
            if missing_types:
                missing = ", ".join(sorted(missing_types))
                raise PlannerError(
                    f"Rule-based planner cannot satisfy '{goal}' with current behavior set. "
                    f"Missing leaf types: {missing}"
                )
            return BehaviorTreeStructure(
                goal=goal,
                description="Detect the target object, then pick it up.",
                root=SequenceNode(
                    type="sequence",
                    name="pickup_sequence",
                    children=[
                        LeafNode(type="detect_object"),
                        LeafNode(type="pick_object"),
                    ],
                ),
            )

        raise ValueError(f"Unknown goal: {goal}")
