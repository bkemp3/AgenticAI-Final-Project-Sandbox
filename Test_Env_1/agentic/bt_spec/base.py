from pydantic import BaseModel, ConfigDict


class BTNodeModel(BaseModel):
    """Shared base model for behavior tree specification nodes.

    The spec layer is intended to be a small validated structure that can be
    produced by rule-based planners now and by LLMs later.
    """

    id: str | None = None
    name: str | None = None

    model_config = ConfigDict(extra="forbid")
