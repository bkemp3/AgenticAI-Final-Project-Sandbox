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
- Reusable planner prompt assets live under `prompts/`. The planner composes these files with behavior-catalog metadata at runtime.

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

Rule-based planner with explicit behavior set:

```bash
uv run examples/demo.py --planner rule_based --behaviors behavior_sets/base.yaml
```

LLM planner:

```bash
uv run examples/demo.py --planner llm
```

Config-driven LLM task runner:

```bash
uv run python examples/llm_task_demo.py configs/llm_collection_run.yaml
```

Timed value-collection simulator demo:

```bash
python3 examples/collection_sim_demo.py
```

This demo loads the YAML environment, advances the simulator by one timestep, shows partial observations, runs a short scripted collection sequence, and prints the resulting event log and final metrics.

## Behavior Sets

- Behavior sets are configured via YAML files under `behavior_sets/` (JSON also supported).
- The default set is `behavior_sets/base.yaml`.
- Each run can select a behavior set using `--behaviors <path>`.
- Behavior sets define:
  - Allowed leaf/condition node types for planning
  - Runtime class mappings used for compilation/execution
  - Optional leaf parameter metadata that the LLM planner may use when generating trees

## Simulation

- `agentic/simulation/` contains a ground-truth collection simulator kept separate from `bt_spec` and `bt_runtime`.
- Environment parameters live in YAML, with a sample at `configs/collection_env.yaml`.
- The simulator tracks hidden object state, stochastic pickup failures, object disappearance, event logs, observations, and task metrics.
- For collection tasks, `behavior_sets/collection.yaml` defines the BT leaves and allowed leaf params available to an LLM planner, while `configs/collection_env.yaml` defines the simulator environment.
- `configs/llm_collection_run.yaml` is the minimal top-level run config for LLM-driven collection experiments. It points to a task adapter, environment YAML, behavior-set YAML, system prompt file, model name, task prompt, and tick budget.
- `configs/llm_collection_run.yaml` also controls whether a high-level environment summary is included in the user prompt and how many times the planner retries after validation failure.
- `prompts/collection_planner_system.txt` stores the collection-specific system prompt referenced by the run config.
- `prompts/planner_system_base.txt` and `prompts/planner_user_base.txt` store the reusable planner prompt text used by `agentic/planning/prompting.py`.
- `examples/collection_compiled_tree_demo.py` shows the two-YAML split by loading the collection behavior set, loading the collection environment, compiling a generated-style tree, and executing it against the simulator.
- `examples/llm_task_demo.py` is a task-agnostic runner. It loads a run config, imports the configured task adapter, composes the final prompts from prompt assets plus behavior metadata, asks the LLM for a tree, executes it against the adapter-provided world, and writes prompts, events, metrics, and the generated tree spec to the configured output directory.
