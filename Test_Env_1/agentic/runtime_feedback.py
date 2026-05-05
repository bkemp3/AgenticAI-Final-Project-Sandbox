from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RuntimeCriticAction(str, Enum):
    CONTINUE = "CONTINUE"
    REQUEST_REPAIR = "REQUEST_REPAIR"


class SuspectedNodeRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str | None = None
    node_name: str | None = None
    node_type: str | None = None
    tree_path: str | None = None


class RuntimeTickTraceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tick: int
    bt_status: str
    summary: str
    active_nodes: list[dict[str, str]] = Field(default_factory=list)


class RuntimeCriticDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: RuntimeCriticAction
    diagnosis: str
    repair_instructions: list[str] = Field(default_factory=list)
    suspected_node: SuspectedNodeRef | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class PlannerRepairRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    diagnosis: str
    repair_instructions: list[str] = Field(default_factory=list)
    suspected_node: SuspectedNodeRef | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    visible_world_observation: Any = None
    recent_events: list[Any] = Field(default_factory=list)
    recent_tick_trace: list[dict[str, Any]] = Field(default_factory=list)
    task_progress_signals: Any = None
    metrics: Any = None
