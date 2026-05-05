from __future__ import annotations

import os
from collections.abc import Callable

from openai import OpenAI

from agentic.behaviors.catalog import BehaviorCatalog
from agentic.bt_spec.tree_structure import BehaviorTreeStructure
from agentic.planning.base import BasePlanner, PlannerError
from agentic.planning.prompting import (
    build_behavior_subset_block,
    build_repair_user_prompt,
    build_retry_user_prompt,
    build_system_prompt,
    build_user_prompt,
)
from agentic.runtime_feedback import PlannerRepairRequest


class LLMPlanner(BasePlanner):
    """Planner that asks an OpenAI model for a structured behavior tree."""

    def __init__(
        self,
        behavior_catalog: BehaviorCatalog,
        model: str = "gpt-4.1-mini",
        client: OpenAI | None = None,
        system_prompt_override: str | None = None,
        user_prompt_override: str | None = None,
        retry_limit: int = 0,
        plan_validator: Callable[[BehaviorTreeStructure], str | None] | None = None,
    ) -> None:
        self.behavior_catalog = behavior_catalog
        self.model = model
        self.client = client
        self.system_prompt_override = system_prompt_override
        self.user_prompt_override = user_prompt_override
        self.retry_limit = retry_limit
        self.plan_validator = plan_validator

    def create_plan(self, goal: str) -> BehaviorTreeStructure:
        return self._create_plan_from_messages(
            [
                {
                    "role": "system",
                    "content": self.system_prompt_override or build_system_prompt(self.behavior_catalog),
                },
                {
                    "role": "user",
                    "content": self.user_prompt_override or build_user_prompt(goal, self.behavior_catalog),
                },
            ]
        )

    def repair_plan(
        self,
        goal: str,
        current_tree: BehaviorTreeStructure,
        repair_request: PlannerRepairRequest,
    ) -> BehaviorTreeStructure:
        base_user_prompt = self.user_prompt_override or build_user_prompt(goal, self.behavior_catalog)
        messages = [
            {
                "role": "system",
                "content": self.system_prompt_override or build_system_prompt(self.behavior_catalog),
            },
            {
                "role": "user",
                "content": build_repair_user_prompt(
                    goal=goal,
                    base_user_prompt=base_user_prompt,
                    current_tree_json=current_tree.model_dump_json(indent=2),
                    diagnosis=repair_request.diagnosis,
                    repair_instructions=repair_request.repair_instructions,
                    suspected_node=repair_request.suspected_node,
                    visible_world_observation=repair_request.visible_world_observation,
                    recent_events=repair_request.recent_events,
                    recent_tick_trace=repair_request.recent_tick_trace,
                    task_progress_signals=repair_request.task_progress_signals,
                    metrics=repair_request.metrics,
                    used_behavior_block=(
                        "Behavior descriptions for nodes used in the current tree:\n"
                        + build_behavior_subset_block(self.behavior_catalog, current_tree)
                    ),
                ),
            },
        ]
        return self._create_plan_from_messages(messages)

    def _create_plan_from_messages(self, messages: list[dict[str, str]]) -> BehaviorTreeStructure:
        last_error: str | None = None

        for attempt in range(self.retry_limit + 1):
            message = self._request_plan(messages)
            parsed = getattr(message, "parsed", None)
            refusal = getattr(message, "refusal", None)

            if refusal:
                raise PlannerError(f"LLM planner refused the request: {refusal}")
            if parsed is None:
                last_error = "LLM planner did not return parsable structured output."
            else:
                try:
                    plan = BehaviorTreeStructure.model_validate(parsed.model_dump())
                except Exception as exc:
                    last_error = f"LLM planner returned invalid plan data: {exc}"
                else:
                    if self.plan_validator is not None:
                        validation_error = self.plan_validator(plan)
                        if validation_error is not None:
                            last_error = f"LLM planner returned a semantically invalid plan: {validation_error}"
                        else:
                            return plan
                    else:
                        return plan

            if attempt < self.retry_limit:
                messages.append(
                    {
                        "role": "user",
                        "content": build_retry_user_prompt(last_error),
                    }
                )

        raise PlannerError(last_error or "LLM planner failed without a specific error.")

    def _request_plan(self, messages: list[dict[str, str]]):
        client = self._get_client()

        try:
            completion = client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=BehaviorTreeStructure,
            )
        except Exception as exc:
            raise PlannerError(f"LLM planner API call failed: {exc}") from exc

        return completion.choices[0].message

    def _get_client(self) -> OpenAI:
        if self.client is not None:
            return self.client

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise PlannerError(
                "OPENAI_API_KEY is not set. Set it to use LLMPlanner."
            )

        self.client = OpenAI(api_key=api_key)
        return self.client
