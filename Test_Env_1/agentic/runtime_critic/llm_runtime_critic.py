from __future__ import annotations

import os

from openai import OpenAI

from agentic.behaviors.catalog import BehaviorCatalog
from agentic.bt_spec.tree_structure import BehaviorTreeStructure
from agentic.runtime_feedback import RuntimeCriticDecision
from agentic.runtime_critic.prompting import (
    build_runtime_critic_system_prompt,
    build_runtime_critic_user_prompt,
    build_used_behavior_block,
)


class RuntimeCriticError(RuntimeError):
    """Raised when the runtime critic cannot produce a valid verdict."""


class LLMRuntimeCritic:
    """LLM-backed runtime critic that inspects partial execution context."""

    def __init__(
        self,
        behavior_catalog: BehaviorCatalog,
        model: str = "gpt-4.1-mini",
        client: OpenAI | None = None,
    ) -> None:
        self.behavior_catalog = behavior_catalog
        self.model = model
        self.client = client

    def assess(
        self,
        *,
        goal: str,
        task_objective: str,
        tree_spec: BehaviorTreeStructure,
        visible_world_observation: object | None,
        recent_events: list[object],
        recent_tick_trace: list[object],
        task_progress_signals: object | None,
        metrics: object | None,
        total_ticks: int,
        repair_count: int,
        bt_status: str | None,
    ) -> RuntimeCriticDecision:
        messages = [
            {
                "role": "system",
                "content": build_runtime_critic_system_prompt(self.behavior_catalog),
            },
            {
                "role": "user",
                "content": build_runtime_critic_user_prompt(
                    goal=goal,
                    task_objective=task_objective,
                    tree_spec=tree_spec,
                    visible_world_observation=visible_world_observation,
                    recent_events=recent_events,
                    recent_tick_trace=recent_tick_trace,
                    task_progress_signals=task_progress_signals,
                    metrics=metrics,
                    used_behavior_block=build_used_behavior_block(self.behavior_catalog, tree_spec),
                    total_ticks=total_ticks,
                    repair_count=repair_count,
                    bt_status=bt_status,
                ),
            },
        ]
        client = self._get_client()

        try:
            completion = client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=RuntimeCriticDecision,
            )
        except Exception as exc:
            raise RuntimeCriticError(f"Runtime critic API call failed: {exc}") from exc

        message = completion.choices[0].message
        refusal = getattr(message, "refusal", None)
        parsed = getattr(message, "parsed", None)
        if refusal:
            raise RuntimeCriticError(f"Runtime critic refused the request: {refusal}")
        if parsed is None:
            raise RuntimeCriticError("Runtime critic did not return parsable structured output.")
        return RuntimeCriticDecision.model_validate(parsed.model_dump())

    def _get_client(self) -> OpenAI:
        if self.client is not None:
            return self.client

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeCriticError("OPENAI_API_KEY is not set. Set it to use the runtime critic.")

        self.client = OpenAI(api_key=api_key)
        return self.client
