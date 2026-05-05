import argparse
import json
import sys
from datetime import datetime
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
from agentic.serialization import serialize_value
from agentic.tasks import load_task_adapter


def main() -> None:
    args = _parse_args()
    config_path = args.config_path
    repo_root = Path(__file__).resolve().parents[1]
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

    output_dir = _prepare_run_output_dir(
        output_root=Path(run_config.output_dir),
        namespace=args.output_namespace or run_config.output_namespace or run_config.goal,
    )
    (output_dir / "system_prompt.txt").write_text(system_prompt + "\n", encoding="utf-8")
    (output_dir / "user_prompt.txt").write_text(user_prompt + "\n", encoding="utf-8")
    (output_dir / "resolved_run_config.json").write_text(
        json.dumps(
            {
                "config_path": _to_project_relative_path(Path(config_path).resolve(), repo_root),
                "output_root": _to_project_relative_path(Path(run_config.output_dir).resolve(), repo_root),
                "output_namespace": args.output_namespace or run_config.output_namespace or run_config.goal,
                "planner_model": run_config.model,
                "critic_model": run_config.critic.model or run_config.model,
                "critic_enabled": run_config.critic.enabled,
                "max_tree_ticks": run_config.max_tree_ticks,
                "critic_max_repairs": run_config.critic.max_repairs,
                "render_grid_each_tick": run_config.logging.render_grid_each_tick,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    app = build_orchestration_app(include_runtime_critic=run_config.critic.enabled)
    graph_artifacts = export_langgraph_visualization(
        app,
        name=f"{run_config.goal}_langgraph",
        output_dir=output_dir / "graphs",
    )

    final_state = app.invoke(
        {
            "goal": run_config.goal,
            "model": run_config.model,
            "task_prompt": run_config.task_prompt,
            "planner_type": run_config.planner_type,
            "world_state": world,
            "behavior_catalog": behavior_catalog,
            "task_adapter": task_adapter,
            "system_prompt_override": system_prompt,
            "user_prompt_override": user_prompt,
            "max_tree_ticks": run_config.max_tree_ticks,
            "retry_limit": run_config.retry_limit,
            "critic_enabled": run_config.critic.enabled,
            "critic_model": run_config.critic.model,
            "critic_interval_ticks": run_config.critic.interval_ticks,
            "critic_max_repairs": run_config.critic.max_repairs,
            "critic_context_window_events": run_config.critic.context_window_events,
            "critic_context_window_ticks": run_config.critic.context_window_ticks,
            "render_grid_each_tick": run_config.logging.render_grid_each_tick,
            "world_trace_path": str(output_dir / "world_trace.txt"),
            "tree_output_dir": str(output_dir / "trees"),
            "planner": None,
            "runtime_critic": None,
            "tree_spec": None,
            "compiled_tree": None,
            "execution_status": None,
            "runtime_bt_status": None,
            "error_message": None,
            "tree_image_path": None,
            "tree_spec_history": [],
            "tree_artifact_history": [],
            "graph_mermaid_path": graph_artifacts["mermaid"],
            "graph_image_path": graph_artifacts["png"],
            "tick_count": 0,
            "repair_count": 0,
            "runtime_tick_trace": [],
            "critic_decision": None,
            "critic_history": [],
            "planner_repair_request": None,
            "critic_due": False,
            "execution_should_continue": False,
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
            json.dumps(_relativize_paths(tree_artifacts, repo_root), indent=2) + "\n",
            encoding="utf-8",
        )

    metrics = task_adapter.get_metrics(world)
    print(f"Final metrics: {metrics}")
    (output_dir / "metrics.json").write_text(
        json.dumps(serialize_value(metrics), indent=2) + "\n",
        encoding="utf-8",
    )
    serialized_events = [serialize_value(event) for event in task_adapter.get_events(world)]
    (output_dir / "events.json").write_text(
        json.dumps(serialized_events, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "runtime_tick_trace.json").write_text(
        json.dumps(serialize_value(final_state.get("runtime_tick_trace") or []), indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "critic_history.json").write_text(
        json.dumps(serialize_value(final_state.get("critic_history") or []), indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "tree_spec_history.json").write_text(
        json.dumps(_relativize_paths(serialize_value(final_state.get("tree_spec_history") or []), repo_root), indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "tree_artifact_history.json").write_text(
        json.dumps(_relativize_paths(serialize_value(final_state.get("tree_artifact_history") or []), repo_root), indent=2) + "\n",
        encoding="utf-8",
    )
    summary = _build_run_summary(
        run_config=run_config,
        config_path=config_path,
        output_dir=output_dir,
        final_state=final_state,
        metrics=metrics,
        repo_root=repo_root,
    )
    (output_dir / "run_summary.json").write_text(
        json.dumps(serialize_value(summary), indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "run_summary.md").write_text(
        _render_run_summary_markdown(summary) + "\n",
        encoding="utf-8",
    )
    print("Event log:")
    for event in task_adapter.get_events(world):
        print(f"  event: {event}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the config-driven LangGraph task demo.")
    parser.add_argument(
        "config_path",
        nargs="?",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "llm_collection_run.yaml"),
        help="Path to the run config YAML.",
    )
    parser.add_argument(
        "--output-namespace",
        default=None,
        help="Optional namespace under outputs/. Overrides output_namespace from config.",
    )
    return parser.parse_args()

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


def _prepare_run_output_dir(*, output_root: Path, namespace: str) -> Path:
    namespace = namespace.strip()
    if not namespace:
        raise ValueError("Output namespace cannot be empty.")

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    namespace_dir = output_root / namespace
    run_dir = namespace_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    latest_path = namespace_dir / "latest"
    if latest_path.exists() or latest_path.is_symlink():
        if latest_path.is_dir() and not latest_path.is_symlink():
            raise RuntimeError(f"Expected latest pointer to be a symlink, found directory: {latest_path}")
        latest_path.unlink()
    latest_path.symlink_to(run_dir.name)
    return run_dir


def _build_run_summary(*, run_config, config_path: str, output_dir: Path, final_state: dict, metrics, repo_root: Path) -> dict[str, object]:
    return {
        "run_timestamp_dir": output_dir.name,
        "run_output_dir": _to_project_relative_path(output_dir, repo_root),
        "namespace": output_dir.parent.name,
        "config_path": _to_project_relative_path(Path(config_path).resolve(), repo_root),
        "planner_type": run_config.planner_type,
        "planner_model": run_config.model,
        "critic_enabled": run_config.critic.enabled,
        "critic_model": run_config.critic.model or run_config.model,
        "goal": run_config.goal,
        "task_prompt": run_config.task_prompt,
        "execution_status": final_state.get("execution_status"),
        "runtime_bt_status": final_state.get("runtime_bt_status"),
        "tick_count": final_state.get("tick_count"),
        "repair_count": final_state.get("repair_count"),
        "critic_decisions": final_state.get("critic_history") or [],
        "final_metrics": metrics,
        "tree_spec_history": _relativize_paths(final_state.get("tree_spec_history") or [], repo_root),
        "tree_artifact_history": _relativize_paths(final_state.get("tree_artifact_history") or [], repo_root),
        "artifact_paths": {
            "system_prompt": _to_project_relative_path(output_dir / "system_prompt.txt", repo_root),
            "user_prompt": _to_project_relative_path(output_dir / "user_prompt.txt", repo_root),
            "tree_spec": _to_project_relative_path(output_dir / "tree_spec.json", repo_root),
            "tree_spec_history": _to_project_relative_path(output_dir / "tree_spec_history.json", repo_root),
            "tree_artifact_history": _to_project_relative_path(output_dir / "tree_artifact_history.json", repo_root),
            "metrics": _to_project_relative_path(output_dir / "metrics.json", repo_root),
            "events": _to_project_relative_path(output_dir / "events.json", repo_root),
            "world_trace": _to_project_relative_path(output_dir / "world_trace.txt", repo_root),
            "runtime_tick_trace": _to_project_relative_path(output_dir / "runtime_tick_trace.json", repo_root),
            "critic_history": _to_project_relative_path(output_dir / "critic_history.json", repo_root),
            "langgraph_mermaid": _to_project_relative_path(final_state.get("graph_mermaid_path"), repo_root),
            "langgraph_png": _to_project_relative_path(final_state.get("graph_image_path"), repo_root),
        },
    }


def _render_run_summary_markdown(summary: dict[str, object]) -> str:
    artifact_paths = summary["artifact_paths"]
    lines = [
        "# Run Summary",
        "",
        f"- Namespace: `{summary['namespace']}`",
        f"- Run directory: `{summary['run_output_dir']}`",
        f"- Config: `{summary['config_path']}`",
        f"- Planner: `{summary['planner_type']}` / `{summary['planner_model']}`",
        f"- Critic: `{summary['critic_enabled']}` / `{summary['critic_model']}`",
        f"- Goal: `{summary['goal']}`",
        f"- Execution status: `{summary['execution_status']}`",
        f"- Runtime BT status: `{summary['runtime_bt_status']}`",
        f"- Tick count: `{summary['tick_count']}`",
        f"- Repairs used: `{summary['repair_count']}`",
        "",
        "## Final Metrics",
        "",
        "```json",
        json.dumps(serialize_value(summary["final_metrics"]), indent=2),
        "```",
        "",
        "## Critic Decisions",
        "",
        "```json",
        json.dumps(serialize_value(summary["critic_decisions"]), indent=2),
        "```",
        "",
        "## Tree History",
        "",
        "```json",
        json.dumps(serialize_value(summary["tree_spec_history"]), indent=2),
        "```",
        "",
        "## Artifacts",
        "",
        f"- System prompt: `{artifact_paths['system_prompt']}`",
        f"- User prompt: `{artifact_paths['user_prompt']}`",
        f"- Tree spec: `{artifact_paths['tree_spec']}`",
        f"- Tree spec history: `{artifact_paths['tree_spec_history']}`",
        f"- Tree artifact history: `{artifact_paths['tree_artifact_history']}`",
        f"- Metrics: `{artifact_paths['metrics']}`",
        f"- Events: `{artifact_paths['events']}`",
        f"- World trace: `{artifact_paths['world_trace']}`",
        f"- Runtime tick trace: `{artifact_paths['runtime_tick_trace']}`",
        f"- Critic history: `{artifact_paths['critic_history']}`",
        f"- LangGraph mermaid: `{artifact_paths['langgraph_mermaid']}`",
        f"- LangGraph PNG: `{artifact_paths['langgraph_png']}`",
    ]
    return "\n".join(lines)


def _relativize_paths(value, repo_root: Path):
    if isinstance(value, dict):
        return {key: _relativize_paths(item, repo_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_relativize_paths(item, repo_root) for item in value]
    if isinstance(value, str):
        return _to_project_relative_path(value, repo_root)
    return value


def _to_project_relative_path(value, repo_root: Path):
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
