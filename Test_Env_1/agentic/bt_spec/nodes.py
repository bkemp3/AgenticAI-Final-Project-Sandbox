from __future__ import annotations

from typing import Literal

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


class ParamEntry(BTNodeModel):
    """Strict parameter entry for LLM structured output compatibility."""

    key: str
    value: str | int | float | bool | None


class LeafNode(BTNodeModel):
    """Dynamic leaf or condition node resolved by runtime behavior registry."""

    type: str
    params: list[ParamEntry] = Field(default_factory=list)


BehaviorTreeNode = (
    SequenceNode
    | SelectorNode
    | LeafNode
)

SequenceNode.model_rebuild()
SelectorNode.model_rebuild()
