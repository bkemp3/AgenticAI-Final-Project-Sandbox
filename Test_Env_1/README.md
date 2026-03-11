# Agentic Sandbox

`agentic-sandbox` is a small Python research sandbox for experimenting with autonomous agents that plan with validated behavior tree specs and execute them with `py_trees`.

The project is intentionally compact, but the architecture is split so it can grow toward LLM planning, monitoring, replanning, and recovery.

## Three Layers

- Planner layer: `RuleBasedPlanner` and `LLMPlanner` generate a validated `BehaviorTreeStructure`.
- `bt_runtime`: compiles that structure into an executable `py_trees` tree and runs it.
- LangGraph orchestration: coordinates the end-to-end pipeline around planning, compilation, visualization, and execution.

Important: LangGraph is not the behavior tree. The behavior tree itself remains a `py_trees` runtime artifact compiled from `bt_spec`.

Current flow:

Goal -> planner selection -> plan generation -> validation -> compilation -> visualization -> execution -> result

This keeps orchestration separate from the behavior tree runtime and prepares the system for future replanning, monitoring, and recovery.

## Planning

- `BasePlanner` defines the planner interface.
- `RuleBasedPlanner` is the current baseline planner.
- `LLMPlanner` uses the OpenAI Python SDK to request a structured `BehaviorTreeStructure`.
- Planners return validated `BehaviorTreeStructure` objects, not executable code.

To use `LLMPlanner`, set:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

## Visualization

- Runtime trees are printed as terminal text.
- The demo also attempts image export with `py_trees`.
- The LangGraph orchestration graph is also exported as Mermaid text and, when available, as a PNG.
- Generated artifacts are stored under `outputs/trees/`.
- LangGraph graph artifacts are stored under `outputs/graphs/`.

## Run

Default rule-based demo:

```bash
uv run examples/demo.py
```

Explicit rule-based planner:

```bash
uv run examples/demo.py --planner rule_based
```

LLM planner:

```bash
uv run examples/demo.py --planner llm
```
