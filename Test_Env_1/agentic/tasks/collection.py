from __future__ import annotations

from dataclasses import dataclass

import py_trees

from agentic.bt_runtime.collection_behaviors import BLACKBOARD_OBSERVATION, BLACKBOARD_SELECTED_ID
from agentic.bt_spec.nodes import LeafNode, SequenceNode
from agentic.bt_spec.tree_structure import BehaviorTreeStructure
from agentic.planning.prompting import summarize_collection_config
from agentic.simulation import CollectionObservation, CollectionWorld, load_collection_config


@dataclass(frozen=True)
class CollectionTaskAdapter:
    def load_world(self, environment_config_path: str) -> CollectionWorld:
        config, objects = load_collection_config(environment_config_path)
        return CollectionWorld(config=config, objects=objects)

    def summarize_world(self, world: CollectionWorld) -> str:
        return summarize_collection_config(world.state.config)

    def describe_tick(self, world: CollectionWorld, tree: py_trees.trees.BehaviourTree, tick: int) -> str:
        blackboard = py_trees.blackboard.Blackboard()
        selected_object = self._safe_blackboard_get(blackboard, f"/{BLACKBOARD_SELECTED_ID}")
        task_success = self.is_success(world)
        return (
            f"tick={tick} status={tree.root.status} selected_object={selected_object} "
            f"position={world.state.robot.position} held={world.state.robot.holding_object_id} "
            f"deposited_value={world.state.robot.inventory_value} task_success={task_success}"
        )

    def render_tick(self, world: CollectionWorld, tree: py_trees.trees.BehaviourTree, tick: int) -> str | None:
        blackboard = py_trees.blackboard.Blackboard()
        selected_object = self._safe_blackboard_get(blackboard, f"/{BLACKBOARD_SELECTED_ID}")
        observation = self._safe_blackboard_get(blackboard, f"/{BLACKBOARD_OBSERVATION}")
        visible_ids = self._visible_object_ids(observation)
        width, height = world.state.config.grid_size
        cell_texts: list[list[str]] = []
        cell_width = 12

        for y in range(height):
            row: list[str] = []
            for x in range(width):
                row.append(self._render_cell(world, position=(x, y), visible_ids=visible_ids).ljust(cell_width))
            cell_texts.append(row)

        held_object = self._held_objects(world)
        deposited_objects = self._deposited_objects(world)
        lines = [
            (
                f"tick {tick} | tree={tree.root.name} | bt_status={tree.root.status.name} "
                f"| selected_target={selected_object}"
            ),
            "",
        ]
        for row in cell_texts:
            lines.append(" | ".join(row))
        lines.extend(
            [
                "",
                f"held: {self._total_value(held_object)} | {self._format_object_list(held_object)}",
                f"deposited: {self._total_value(deposited_objects)} | {self._format_object_list(deposited_objects)}",
            ]
        )
        return "\n".join(lines)

    def get_visible_observation(self, world: CollectionWorld, tree: py_trees.trees.BehaviourTree) -> object | None:
        blackboard = py_trees.blackboard.Blackboard()
        return self._safe_blackboard_get(blackboard, f"/{BLACKBOARD_OBSERVATION}")

    def get_events(self, world: CollectionWorld) -> list[object]:
        return list(world.events)

    def get_metrics(self, world: CollectionWorld) -> object:
        metrics = world.get_metrics()
        return {
            "simulator_metrics": metrics,
            "task_success": self.is_success(world),
            "target_value_goal_met": world.state.robot.inventory_value >= world.state.config.target_value,
            "target_value": world.state.config.target_value,
        }

    def get_progress_signals(self, world: CollectionWorld) -> object:
        return {
            "task_success": self.is_success(world),
            "inventory_value": world.state.robot.inventory_value,
            "target_value": world.state.config.target_value,
            "inventory_weight": world.state.robot.inventory_weight,
            "carry_capacity_remaining": world.remaining_carry_capacity(),
            "holding_object_id": world.state.robot.holding_object_id,
            "time_remaining": max(world.state.config.max_timesteps - world.state.timestep, 0),
            "timestep": world.state.timestep,
        }

    def is_success(self, world: CollectionWorld) -> bool:
        return world.state.robot.inventory_value >= world.state.config.target_value

    def is_terminal(self, world: CollectionWorld) -> bool:
        return self.is_success(world) or world.state.timestep >= world.state.config.max_timesteps

    def validate_plan(self, tree_spec: BehaviorTreeStructure) -> str | None:
        root = tree_spec.root
        if isinstance(root, SequenceNode):
            for child in root.children:
                if isinstance(child, LeafNode) and child.type == "collection_value_goal_met":
                    return (
                        "collection_value_goal_met cannot be a direct child of the top-level sequence because it "
                        "fails until the goal is already satisfied and prevents all collection work from running. "
                        "Use a top-level selector with the goal check as one branch and the work loop as another."
                    )
                if not isinstance(child, LeafNode):
                    break
        return None

    @staticmethod
    def _safe_blackboard_get(blackboard: py_trees.blackboard.Blackboard, key: str):
        try:
            return blackboard.get(key)
        except KeyError:
            return None

    def _render_cell(
        self,
        world: CollectionWorld,
        *,
        position: tuple[int, int],
        visible_ids: set[str],
    ) -> str:
        tokens: list[str] = []
        if world.state.robot.position == position:
            tokens.append("A")
        if world.state.config.dropoff_location == position:
            tokens.append("D")

        for obj in sorted(world.state.objects.values(), key=lambda item: item.object_id):
            if not obj.available or obj.position != position:
                continue
            tokens.append(self._object_token(obj.object_id, obj.value, obj.object_id in visible_ids))
        return ",".join(tokens) if tokens else "."

    def _object_token(self, object_id: str, value: float, visible: bool) -> str:
        base = f"{object_id[0].lower()}-{int(value) if float(value).is_integer() else value}"
        return f"({base})" if visible else base

    def _visible_object_ids(self, observation: CollectionObservation | None) -> set[str]:
        if observation is None:
            return set()
        return {obj.object_id for obj in observation.visible_objects}

    def _held_objects(self, world: CollectionWorld):
        held_id = world.state.robot.holding_object_id
        if held_id is None:
            return []
        held = world.state.objects.get(held_id)
        return [held] if held is not None else []

    def _deposited_objects(self, world: CollectionWorld):
        held_id = world.state.robot.holding_object_id
        deposited = []
        for obj in sorted(world.state.objects.values(), key=lambda item: item.object_id):
            if obj.collected_by == "robot" and obj.object_id != held_id:
                deposited.append(obj)
        return deposited

    def _format_object_list(self, objects) -> str:
        if not objects:
            return "[]"
        return "[" + ", ".join(self._object_token(obj.object_id, obj.value, False) for obj in objects) + "]"

    def _total_value(self, objects) -> str:
        total = sum(float(obj.value) for obj in objects)
        return str(int(total) if float(total).is_integer() else total)


adapter = CollectionTaskAdapter()
