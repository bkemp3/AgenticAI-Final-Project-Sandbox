from pathlib import Path

from agentic.behaviors.catalog import BehaviorCatalog
from agentic.simulation import CollectionConfig


PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "planner_system_base.txt"
USER_PROMPT_PATH = PROMPTS_DIR / "planner_user_base.txt"


def build_system_prompt(catalog: BehaviorCatalog) -> str:
    """Build the planner system prompt from prompt assets plus behavior metadata."""

    base = _load_prompt_asset(SYSTEM_PROMPT_PATH)
    behavior_block = build_behavior_catalog_block(catalog)
    return "\n\n".join([base, behavior_block]).strip()


def _format_behavior_rule(behavior) -> str:
    base = f"- {behavior.type}: leaf/condition, no children. {behavior.description}".strip()
    if not behavior.params:
        return base
    params = "; ".join(f"{param.key}: {param.description}" for param in behavior.params)
    return f"{base} Allowed params: {params}"


def build_user_prompt(goal: str, catalog: BehaviorCatalog) -> str:
    """Build the planner user prompt from prompt assets."""

    allowed_types = ", ".join(catalog.allowed_node_types)
    return _load_prompt_asset(USER_PROMPT_PATH).format(
        goal=goal,
        allowed_types=allowed_types,
    ).strip()


def build_behavior_catalog_block(catalog: BehaviorCatalog) -> str:
    """Describe the allowed BT nodes and leaf params for planner prompts."""

    leaf_rules = "\n".join(_format_behavior_rule(behavior) for behavior in catalog.leaf_behaviors)
    allowed_types = ", ".join(catalog.allowed_node_types)
    return "\n".join(
        [
            f"You may only use these node types: {allowed_types}.",
            "",
            "Available leaf/condition nodes:",
            leaf_rules,
        ]
    ).strip()


def build_run_system_prompt(base_system_prompt: str, catalog: BehaviorCatalog) -> str:
    """Compose a run-specific system prompt from prompt assets and behavior metadata."""

    return "\n\n".join([base_system_prompt.strip(), build_behavior_catalog_block(catalog)]).strip()


def build_run_user_prompt(
    *,
    goal: str,
    task_prompt: str,
    catalog: BehaviorCatalog,
    environment_summary: str | None = None,
) -> str:
    """Compose a run-specific user prompt from the task prompt and optional environment summary."""

    base = build_user_prompt(goal, catalog)
    sections = [base, f"Task instruction:\n{task_prompt.strip()}"]
    if environment_summary:
        sections.append(f"High-level environment summary:\n{environment_summary.strip()}")
    return "\n\n".join(sections).strip()


def summarize_collection_config(config: CollectionConfig) -> str:
    """Produce a high-level environment summary without revealing hidden object state."""

    return "\n".join(
        [
            f"- grid_size: {config.grid_size}",
            f"- target_value: {config.target_value}",
            f"- carry_capacity: {config.carry_capacity}",
            f"- max_timesteps: {config.max_timesteps}",
            f"- dropoff_location: {config.dropoff_location}",
            "- hidden state remains unknown to the planner",
            "- objects may disappear and pickup can fail stochastically",
            "- deposited value is what counts toward success",
        ]
    )


def _load_prompt_asset(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()
