from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from agentic.bt_spec.base import BTNodeModel


class SequenceNode(BTNodeModel):
    """Composite node that runs children in order."""

    type: Literal["sequence"]
    children: list["BehaviorTreeNode"] = Field(min_length=1)


class SelectorNode(BTNodeModel):
    """Composite node that tries children until one succeeds."""

    type: Literal["selector"]
    children: list["BehaviorTreeNode"] = Field(min_length=1)


class DetectObjectNode(BTNodeModel):
    """Leaf node for detecting an object in the environment."""

    type: Literal["detect_object"]


class PickObjectNode(BTNodeModel):
    """Leaf node for picking up an object."""

    type: Literal["pick_object"]
    target: str | None = None


class ObjectVisibleNode(BTNodeModel):
    """Condition node that checks whether an object is visible."""

    type: Literal["object_visible"]


class HoldingObjectNode(BTNodeModel):
    """Condition node that checks whether the robot holds an object."""

    type: Literal["holding_object"]


BehaviorTreeNode = Annotated[
    SequenceNode
    | SelectorNode
    | DetectObjectNode
    | PickObjectNode
    | ObjectVisibleNode
    | HoldingObjectNode,
    Field(discriminator="type"),
]

SequenceNode.model_rebuild()
SelectorNode.model_rebuild()
