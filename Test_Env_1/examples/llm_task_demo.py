import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

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
)
from agentic.planning.run_config import load_llm_task_run_config
from agentic.tasks import load_task_adapter


def main() -> None:
    config_path = _parse_config_path()
    run_config = load_llm_task_run_config(config_path)
    task_adapter = load_task_adapter(run_config.task_adapter)
    behavior_catalog = load_behavior_catalog(run_config.behavior_set_path)
    world = task_adapter.load_world(run_config.environment_config_path)
    environment_summary = task_adapter.summarize_world(world)

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
        plan_validator=task_adapter.validate_plan,
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
        if task_adapter.is_terminal(world):
            break
        tree.tick()
        print(task_adapter.describe_tick(world, tree, tick))
        events = task_adapter.get_events(world)
        for event in events[printed_events:]:
            print(f"  event: {event}")
        printed_events = len(events)

    metrics = task_adapter.get_metrics(world)
    print(f"Final metrics: {metrics}")
    (output_dir / "metrics.json").write_text(json.dumps(_serialize_value(metrics), indent=2) + "\n", encoding="utf-8")
    serialized_events = [_serialize_value(event) for event in task_adapter.get_events(world)]
    (output_dir / "events.json").write_text(json.dumps(serialized_events, indent=2) + "\n", encoding="utf-8")


def _parse_config_path() -> str:
    if len(sys.argv) > 2 or (len(sys.argv) == 2 and sys.argv[1] in {"-h", "--help"}):
        print("Usage: uv run python examples/llm_task_demo.py [path/to/run_config.yaml]", file=sys.stderr)
        raise SystemExit(2)
    if len(sys.argv) == 2:
        return sys.argv[1]
    return str(Path(__file__).resolve().parents[1] / "configs" / "llm_collection_run.yaml")


def _serialize_value(value):
    if is_dataclass(value):
        return asdict(value)
    return value


if __name__ == "__main__":
    main()
