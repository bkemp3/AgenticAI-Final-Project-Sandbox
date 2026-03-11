# Agentic Sandbox

`agentic-sandbox` is a minimal Python research sandbox for experimenting with an autonomous agent that composes and executes behavior trees.

The code is intentionally small and readable so it can grow later into a larger agentic AI project with components such as LLM planners, runtime monitors, or simulators.

## Folder layout

```text
agentic_sandbox/
├── pyproject.toml
├── README.md
├── agentic/
│   ├── __init__.py
│   ├── agent.py
│   ├── world_state.py
│   ├── bt_spec/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── nodes.py
│   │   └── tree_structure.py
│   ├── planning/
│   │   ├── __init__.py
│   │   └── planner.py
│   ├── behavior_tree/
│   │   ├── __init__.py
│   │   ├── nodes.py
│   │   ├── status.py
│   │   └── tree.py
│   ├── runtime/
│   │   ├── __init__.py
│   │   └── executor.py
│   └── skills/
│       ├── __init__.py
│       ├── base_skill.py
│       └── basic_skills.py
└── examples/
    └── demo.py
```

## Architecture

The sandbox follows a simple flow:

Goal -> Planner -> Behavior Tree -> Skills -> World State

- The agent receives a goal.
- The planner turns that goal into a behavior tree.
- The behavior tree executes skills in order.
- Skills read and update the world state.

## Spec vs Runtime

The project now has a small architecture split:

- `bt_spec` contains the validated, structured behavior tree representation intended for planning and future LLM-generated output.
- The runtime behavior tree remains the execution layer used by the current demo.

Current flow:

Goal -> Planner -> `bt_spec` structure -> runtime tree -> Skills -> World State

## Run

Use `uv` to run the demo:

```bash
uv run examples/demo.py
```
