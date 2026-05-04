from textwrap import dedent

from agentic.behaviors.catalog import BehaviorCatalog

def build_system_prompt(catalog: BehaviorCatalog) -> str:
    """Build the compact planner instructions for the LLM."""

    leaf_rules = "\n".join(
        f"- {behavior.type}: leaf/condition, no children. {behavior.description}".strip()
        for behavior in catalog.leaf_behaviors
    )
    allowed_types = ", ".join(catalog.allowed_node_types)
    return dedent(
        f"""
        You are a behavior tree planner.
        Return a valid BehaviorTreeStructure for the given goal.
        You may only use these node types: {allowed_types}.

        Node rules:
        - sequence: composite, requires one or more children
        - selector: composite, requires one or more children
        {leaf_rules}

        Constraints:
        - Do not invent new node types or extra fields.
        - Composite nodes must have children.
        - Leaf and condition nodes must not have children.
        - Leaf and condition nodes may optionally include params: object.
        - Return a concise plan that matches the goal.
        - Do not generate code.
        """
    ).strip()


def build_user_prompt(goal: str, catalog: BehaviorCatalog) -> str:
    """Build the user prompt for a specific planning request."""

    allowed_types = ", ".join(catalog.allowed_node_types)
    return dedent(
        f"""
        Goal: {goal}

        Build a BehaviorTreeStructure using only the allowed node types:
        {allowed_types}
        """
    ).strip()
