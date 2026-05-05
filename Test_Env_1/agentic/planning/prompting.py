from pathlib import Path

from agentic.behaviors.catalog import BehaviorCatalog


PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "planner_system_base.txt"
USER_PROMPT_PATH = PROMPTS_DIR / "planner_user_base.txt"


def build_system_prompt(catalog: BehaviorCatalog) -> str:
    """Build the planner system prompt from prompt assets plus behavior metadata."""

    leaf_rules = "\n".join(_format_behavior_rule(behavior) for behavior in catalog.leaf_behaviors)
    allowed_types = ", ".join(catalog.allowed_node_types)
    base = _load_prompt_asset(SYSTEM_PROMPT_PATH)
    behavior_block = "\n".join(
        [
            f"You may only use these node types: {allowed_types}.",
            "",
            "Available leaf/condition nodes:",
            leaf_rules,
        ]
    )
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


def _load_prompt_asset(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()
