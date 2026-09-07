import logging
from llm.provider import get_llm
from schema.feature_planner import FeatureEngineeringPlan
from prompts.feature_eng_prompt import system_prompt as FEATURE_ENGINEERING_PLANNER_PROMPT
from utilis.llm_plan import invoke_with_repair
from utilis.constraints import format_constraints

logger = logging.getLogger(__name__)


class FeatureEngineeringPlannerAgent:
    def __init__(self):
        self.llm = get_llm().with_structured_output(FeatureEngineeringPlan)

    async def plan(
        self,
        user_query: str,
        dataset_summary,
        eda_report: dict,
        target_column: str | None = None,
        constraints: list[str] | None = None,
    ) -> FeatureEngineeringPlan:
        logger.info("Starting feature engineering planning")

        target_section = (
            f"\nTarget Column (DO NOT modify, encode, drop, or reference in ANY step): {target_column}"
            if target_column else ""
        )

        user_content = f"""User Query: {user_query}

Dataset Summary: {dataset_summary}

EDA Report: {eda_report}{target_section}{format_constraints(constraints)}

Generate the feature engineering plan now."""

        messages = [
            {"role": "system", "content": FEATURE_ENGINEERING_PLANNER_PROMPT},
            {"role": "user", "content": user_content},
        ]

        response = await invoke_with_repair(self.llm, messages)

        logger.info(
            f"Feature engineering plan generated: {len(response.steps)} steps")

        return response
