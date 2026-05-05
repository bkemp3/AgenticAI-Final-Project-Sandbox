import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

import py_trees
import yaml

# Allow running the demo directly from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic.behaviors import load_behavior_catalog
from agentic.bt_runtime.compiler import compile_behavior_tree
from agentic.bt_runtime.visualization import export_tree_image, print_ascii_tree
from agentic.planning.llm_planner import LLMPlanner
from agentic.planning.prompting import (
    SYSTEM_PROMPT_PATH,
    build_run_system_prompt,
    build_run_user_prompt,
    summarize_collection_config,
)
from agentic.planning.run_config import load_llm_task_run_config
from agentic.simulation import CollectionWorld, load_collection_config


def main() -> None:
    config_path = _parse_config_path()
    run_config = load_llm_task_run_config(config_path)
    behavior_catalog = load_behavior_catalog(run_config.behavior_set_path)
    world, environment_summary = _load_world_and_summary(run_config.environment_config_path)

    base_system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    task_system_prompt = Path(run_config.system_prompt_path).read_text(encoding="utf-8").strip()
    system_prompt = build_run_system_prompt(
        base_system_prompt="\n\n".join([base_system_prompt, task_system_prompt]).strip(),
        catalog=behavior_catalog,
    )
    user_prompt = build_run_user_prompt(
        goal=run_config.goal,
        task_prompt=run_config.task_prompt,
        catalog=behavior_catalog,
        environment_summary=environment_summary if run_config.user_prompt_include_environment_summary else None,
    )

    planner = LLMPlanner(
        behavior_catalog=behavior_catalog,
        model=run_config.model,
        system_prompt_override=system_prompt,
        user_prompt_override=user_prompt,
        retry_limit=run_config.retry_limit,
    )
    tree_spec = planner.create_plan(run_config.goal)
    tree = compile_behavior_tree(tree_spec, world, behavior_catalog.runtime_registry())

    output_dir = Path(run_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "system_prompt.txt").write_text(system_prompt + "\n", encoding="utf-8")
    (output_dir / "user_prompt.txt").write_text(user_prompt + "\n", encoding="utf-8")
    (output_dir / "tree_spec.json").write_text(tree_spec.model_dump_json(indent=2) + "\n", encoding="utf-8")
    tree_artifacts = export_tree_image(
        tree,
        name=run_config.goal,
        output_dir=output_dir / "trees",
    )
    if tree_artifacts:
        (output_dir / "tree_artifacts.json").write_text(
            json.dumps(tree_artifacts, indent=2) + "\n",
            encoding="utf-8",
        )

    print(f"Run config: {run_config}")
    print("System prompt:")
    print(system_prompt)
    print("User prompt:")
    print(user_prompt)
    print("Generated tree spec:")
    print(tree_spec.model_dump_json(indent=2))
    print("Compiled behavior tree:")
    print_ascii_tree(tree)
    if tree_artifacts:
        print(f"Tree image artifacts: {tree_artifacts}")

    printed_events = 0
    for tick in range(run_config.max_tree_ticks):
        if world.is_terminal():
            break
        tree.tick()
        blackboard = py_trees.blackboard.Blackboard()
        selected_object = _safe_blackboard_get(blackboard, "/selected_object_id")
        print(
            f"tick={tick} status={tree.root.status} selected_object={selected_object} "
            f"position={world.state.robot.position} held={world.state.robot.holding_object_id}"
        )
        for event in world.events[printed_events:]:
            print(f"  event: {event}")
        printed_events = len(world.events)

    metrics = world.get_metrics()
    print(f"Final metrics: {metrics}")
    (output_dir / "metrics.json").write_text(json.dumps(asdict(metrics), indent=2) + "\n", encoding="utf-8")
    serialized_events = [_serialize_event(event) for event in world.events]
    (output_dir / "events.json").write_text(json.dumps(serialized_events, indent=2) + "\n", encoding="utf-8")


def _parse_config_path() -> str:
    if len(sys.argv) > 2 or (len(sys.argv) == 2 and sys.argv[1] in {"-h", "--help"}):
        print("Usage: uv run python examples/llm_task_demo.py [configs/llm_collection_run.yaml]", file=sys.stderr)
        raise SystemExit(2)
    if len(sys.argv) == 2:
        return sys.argv[1]
    return str(Path(__file__).resolve().parents[1] / "configs" / "llm_collection_run.yaml")


def _load_world_and_summary(environment_config_path: str) -> tuple[CollectionWorld, str]:
    raw_config = _load_yaml(environment_config_path)
    if _is_collection_environment(raw_config):
        config, objects = load_collection_config(environment_config_path)
        return CollectionWorld(config=config, objects=objects), summarize_collection_config(config)
    raise ValueError(
        "Unsupported environment config for llm_task_demo. "
        "Currently only collection-task environments are supported."
    )


def _load_yaml(path: str) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _is_collection_environment(raw_config: dict[str, object]) -> bool:
    required_keys = {"grid_size", "target_weight", "target_value", "carry_capacity", "max_timesteps", "dropoff_location"}
    return required_keys.issubset(raw_config)


def _safe_blackboard_get(blackboard: py_trees.blackboard.Blackboard, key: str):
    try:
        return blackboard.get(key)
    except KeyError:
        return None


def _serialize_event(event):
    if is_dataclass(event):
        return asdict(event)
    return event


if __name__ == "__main__":
    main()
