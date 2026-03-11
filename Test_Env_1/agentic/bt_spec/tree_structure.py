from agentic.bt_spec.base import BTNodeModel
from agentic.bt_spec.nodes import BehaviorTreeNode


class BehaviorTreeStructure(BTNodeModel):
    """Top-level validated behavior tree structure returned by planning."""

    root: BehaviorTreeNode
    goal: str
    description: str | None = None
