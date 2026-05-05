from __future__ import annotations

from pathlib import Path

import yaml

from agentic.simulation.models import CollectionConfig, GridPosition, MaterialObject


class CollectionConfigError(ValueError):
    """Raised when the collection environment YAML is invalid."""


def load_collection_config(path: str) -> tuple[CollectionConfig, list[MaterialObject]]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    grid_size = _parse_position(raw_config.get("grid_size"), field_name="grid_size")
    dropoff_location = _parse_position(
        raw_config.get("dropoff_location"), field_name="dropoff_location"
    )
    config = CollectionConfig(
        grid_size=grid_size,
        target_weight=float(_require(raw_config, "target_weight")),
        max_timesteps=int(_require(raw_config, "max_timesteps")),
        dropoff_location=dropoff_location,
        disappear_prob=_parse_probability(raw_config.get("disappear_prob"), "disappear_prob"),
        seed=int(raw_config["seed"]) if raw_config.get("seed") is not None else None,
    )

    default_pickup_failure_prob = _parse_probability(
        raw_config.get("default_pickup_failure_prob", 0.0),
        "default_pickup_failure_prob",
    )
    objects = _load_objects(raw_config.get("objects", []), default_pickup_failure_prob)
    return config, objects


def _load_objects(
    raw_objects: list[dict[str, object]], default_pickup_failure_prob: float
) -> list[MaterialObject]:
    if not isinstance(raw_objects, list):
        raise CollectionConfigError("'objects' must be a list when provided.")

    objects: list[MaterialObject] = []
    for index, raw_object in enumerate(raw_objects, start=1):
        if not isinstance(raw_object, dict):
            raise CollectionConfigError(f"Object entry {index} must be a mapping.")
        object_id = str(raw_object.get("object_id") or f"object_{index}")
        objects.append(
            MaterialObject(
                object_id=object_id,
                position=_parse_position(raw_object.get("position"), field_name=f"objects[{index}].position"),
                weight=float(_require(raw_object, "weight", prefix=f"objects[{index}]")),
                value=float(_require(raw_object, "value", prefix=f"objects[{index}]")),
                available=bool(raw_object.get("available", True)),
                pickup_failure_prob=_parse_probability(
                    raw_object.get("pickup_failure_prob", default_pickup_failure_prob),
                    f"objects[{index}].pickup_failure_prob",
                ),
                collected_by=str(raw_object["collected_by"]) if raw_object.get("collected_by") else None,
            )
        )
    return objects


def _require(raw_config: dict[str, object], key: str, prefix: str | None = None) -> object:
    if key not in raw_config:
        if prefix is None:
            raise CollectionConfigError(f"Missing required config field '{key}'.")
        raise CollectionConfigError(f"Missing required config field '{prefix}.{key}'.")
    return raw_config[key]


def _parse_position(value: object, field_name: str) -> GridPosition:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise CollectionConfigError(f"'{field_name}' must be a two-item coordinate.")
    x, y = value
    return int(x), int(y)


def _parse_probability(value: object, field_name: str) -> float:
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise CollectionConfigError(f"'{field_name}' must be between 0.0 and 1.0.")
    return probability
