from __future__ import annotations

import os

from openai import OpenAI

from agentic.behaviors.catalog import BehaviorCatalog
from agentic.bt_spec.tree_structure import BehaviorTreeStructure
from agentic.planning.base import BasePlanner, PlannerError
from agentic.planning.prompting import build_system_prompt, build_user_prompt


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
    ) -> None:
        self.behavior_catalog = behavior_catalog
        self.model = model
        self.client = client
        self.system_prompt_override = system_prompt_override
        self.user_prompt_override = user_prompt_override
        self.retry_limit = retry_limit

    def create_plan(self, goal: str) -> BehaviorTreeStructure:
        messages = [
            {
                "role": "system",
                "content": self.system_prompt_override or build_system_prompt(self.behavior_catalog),
            },
            {
                "role": "user",
                "content": self.user_prompt_override or build_user_prompt(goal, self.behavior_catalog),
            },
        ]
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
                    return BehaviorTreeStructure.model_validate(parsed.model_dump())
                except Exception as exc:
                    last_error = f"LLM planner returned invalid plan data: {exc}"

            if attempt < self.retry_limit:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous behavior tree was invalid. "
                            f"Validation error: {last_error}. "
                            "Return a corrected BehaviorTreeStructure using only the allowed node types and params."
                        ),
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
