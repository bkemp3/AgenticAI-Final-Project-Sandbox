from __future__ import annotations

from typing import Any, Literal

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


class LeafNode(BTNodeModel):
    """Dynamic leaf or condition node resolved by runtime behavior registry."""

    type: str
    params: dict[str, Any] = Field(default_factory=dict)


BehaviorTreeNode = (
    SequenceNode
    | SelectorNode
    | LeafNode
)

SequenceNode.model_rebuild()
SelectorNode.model_rebuild()
