from textwrap import dedent


ALLOWED_NODE_TYPES = [
    "sequence",
    "selector",
    "detect_object",
    "pick_object",
    "object_visible",
    "holding_object",
]


def build_system_prompt() -> str:
    """Build the compact planner instructions for the LLM."""

    return dedent(
        """
        You are a behavior tree planner.
        Return a valid BehaviorTreeStructure for the given goal.
        You may only use these node types: sequence, selector, detect_object,
        pick_object, object_visible, holding_object.

        Node rules:
        - sequence: composite, requires one or more children
        - selector: composite, requires one or more children
        - detect_object: leaf, no children
        - pick_object: leaf, no children, optional field target: string | null
        - object_visible: condition, no children
        - holding_object: condition, no children

        Constraints:
        - Do not invent new node types or extra fields.
        - Composite nodes must have children.
        - Leaf and condition nodes must not have children.
        - Return a concise plan that matches the goal.
        - Do not generate code.
        """
    ).strip()


def build_user_prompt(goal: str) -> str:
    """Build the user prompt for a specific planning request."""

    allowed_types = ", ".join(ALLOWED_NODE_TYPES)
    return dedent(
        f"""
        Goal: {goal}

        Build a BehaviorTreeStructure using only the allowed node types:
        {allowed_types}
        """
    ).strip()
