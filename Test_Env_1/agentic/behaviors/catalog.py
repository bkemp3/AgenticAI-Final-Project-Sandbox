from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

import py_trees
import yaml


@dataclass(frozen=True)
class BehaviorDefinition:
    type: str
    description: str
    runtime_class: str


@dataclass(frozen=True)
class BehaviorCatalog:
    name: str
    leaf_behaviors: tuple[BehaviorDefinition, ...]

    @property
    def leaf_types(self) -> list[str]:
        return [behavior.type for behavior in self.leaf_behaviors]

    @property
    def allowed_node_types(self) -> list[str]:
        return ["sequence", "selector", *self.leaf_types]

    def runtime_registry(self) -> dict[str, type[py_trees.behaviour.Behaviour]]:
        registry: dict[str, type[py_trees.behaviour.Behaviour]] = {}
        for behavior in self.leaf_behaviors:
            registry[behavior.type] = _load_behaviour_class(behavior.runtime_class)
        return registry


def load_behavior_catalog(path: str | None = None) -> BehaviorCatalog:
    catalog_path = _resolve_catalog_path(path)
    with catalog_path.open("r", encoding="utf-8") as f:
        if catalog_path.suffix.lower() in {".yaml", ".yml"}:
            raw = yaml.safe_load(f)
        elif catalog_path.suffix.lower() == ".json":
            raw = yaml.safe_load(f)
        else:
            raise ValueError(
                f"Unsupported behavior catalog file extension: {catalog_path.suffix}. "
                "Use .yaml, .yml, or .json."
            )

    behaviors = tuple(
        BehaviorDefinition(
            type=item["type"],
            description=item.get("description", ""),
            runtime_class=item["runtime_class"],
        )
        for item in raw["leaf_behaviors"]
    )
    return BehaviorCatalog(name=raw.get("name", catalog_path.stem), leaf_behaviors=behaviors)


def _resolve_catalog_path(path: str | None) -> Path:
    if path:
        return Path(path).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "behavior_sets" / "base.yaml"


def _load_behaviour_class(import_path: str) -> type[py_trees.behaviour.Behaviour]:
    module_path, _, class_name = import_path.rpartition(".")
    if not module_path or not class_name:
        raise ValueError(f"Invalid runtime_class path: {import_path}")

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ValueError(f"Runtime behaviour class not found: {import_path}")
    if not issubclass(cls, py_trees.behaviour.Behaviour):
        raise TypeError(f"Runtime class is not a py_trees Behaviour: {import_path}")
    return cls
