from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel


def serialize_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_value(item) for item in value]
    return value


def to_pretty_json(value: Any) -> str:
    return json.dumps(serialize_value(value), indent=2, sort_keys=True)
