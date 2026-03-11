from __future__ import annotations

import os

from openai import OpenAI

from agentic.bt_spec.tree_structure import BehaviorTreeStructure
from agentic.planning.base import BasePlanner, PlannerError
from agentic.planning.prompting import build_system_prompt, build_user_prompt


class LLMPlanner(BasePlanner):
    """Planner that asks an OpenAI model for a structured behavior tree."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        client: OpenAI | None = None,
    ) -> None:
        self.model = model
        self.client = client

    def create_plan(self, goal: str) -> BehaviorTreeStructure:
        message = self._request_plan(goal)
        parsed = getattr(message, "parsed", None)
        refusal = getattr(message, "refusal", None)

        if refusal:
            raise PlannerError(f"LLM planner refused the request: {refusal}")
        if parsed is None:
            raise PlannerError("LLM planner did not return parsable structured output.")

        try:
            return BehaviorTreeStructure.model_validate(parsed.model_dump())
        except Exception as exc:
            raise PlannerError(f"LLM planner returned invalid plan data: {exc}") from exc

    def _request_plan(self, goal: str):
        client = self._get_client()

        try:
            completion = client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": build_system_prompt()},
                    {"role": "user", "content": build_user_prompt(goal)},
                ],
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
