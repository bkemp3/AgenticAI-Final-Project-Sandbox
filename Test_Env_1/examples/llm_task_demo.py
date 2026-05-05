import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

# Allow running the demo directly from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic.behaviors import load_behavior_catalog
from agentic.orchestration.graph import build_orchestration_app
from agentic.orchestration.visualization import export_langgraph_visualization
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

    output_dir = Path(run_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "system_prompt.txt").write_text(system_prompt + "\n", encoding="utf-8")
    (output_dir / "user_prompt.txt").write_text(user_prompt + "\n", encoding="utf-8")

    app = build_orchestration_app()
    graph_artifacts = export_langgraph_visualization(
        app,
        name=f"{run_config.goal}_langgraph",
        output_dir=output_dir / "graphs",
    )

    final_state = app.invoke(
        {
            "goal": run_config.goal,
            "planner_type": run_config.planner_type,
            "world_state": world,
            "behavior_catalog": behavior_catalog,
            "task_adapter": task_adapter,
            "system_prompt_override": system_prompt,
            "user_prompt_override": user_prompt,
            "max_tree_ticks": run_config.max_tree_ticks,
            "retry_limit": run_config.retry_limit,
            "tree_output_dir": str(output_dir / "trees"),
            "planner": None,
            "tree_spec": None,
            "compiled_tree": None,
            "execution_status": None,
            "error_message": None,
            "tree_image_path": None,
            "graph_mermaid_path": graph_artifacts["mermaid"],
            "graph_image_path": graph_artifacts["png"],
        }
    )
    if final_state.get("error_message"):
        raise RuntimeError(final_state["error_message"])

    tree_spec = final_state.get("tree_spec")
    if tree_spec is None:
        raise RuntimeError("Orchestration finished without a tree spec.")
    (output_dir / "tree_spec.json").write_text(tree_spec.model_dump_json(indent=2) + "\n", encoding="utf-8")

    print(f"Run config: {run_config}")
    print("System prompt:")
    print(system_prompt)
    print("User prompt:")
    print(user_prompt)
    print("Generated tree spec:")
    print(tree_spec.model_dump_json(indent=2))
    print(f"Execution status: {final_state.get('execution_status')}")
    print(f"LangGraph artifacts: {graph_artifacts}")

    tree_artifacts = _collect_tree_artifacts(final_state.get("tree_image_path"))
    if tree_artifacts:
        print(f"Tree image artifacts: {tree_artifacts}")
        (output_dir / "tree_artifacts.json").write_text(
            json.dumps(tree_artifacts, indent=2) + "\n",
            encoding="utf-8",
        )

    metrics = task_adapter.get_metrics(world)
    print(f"Final metrics: {metrics}")
    (output_dir / "metrics.json").write_text(
        json.dumps(_serialize_value(metrics), indent=2) + "\n",
        encoding="utf-8",
    )
    serialized_events = [_serialize_value(event) for event in task_adapter.get_events(world)]
    (output_dir / "events.json").write_text(
        json.dumps(serialized_events, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Event log:")
    for event in task_adapter.get_events(world):
        print(f"  event: {event}")


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


def _collect_tree_artifacts(tree_image_path: str | None) -> dict[str, str] | None:
    if tree_image_path is None:
        return None
    artifact_path = Path(tree_image_path)
    stem = artifact_path.stem
    directory = artifact_path.parent
    artifacts: dict[str, str] = {}
    for suffix in ("dot", "svg", "png"):
        candidate = directory / f"{stem}.{suffix}"
        if candidate.exists():
            artifacts[suffix] = str(candidate)
    return artifacts or None


if __name__ == "__main__":
    main()
