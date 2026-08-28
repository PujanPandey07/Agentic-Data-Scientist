from llm.provider import get_llm
from schema.feature_planner import FeatureEngineeringPlan
from prompts.feature_eng_prompt import system_prompt as FEATURE_ENGINEERING_PLANNER_PROMPT


class FeatureEngineeringPlannerAgent:
    def __init__(self):
        self.llm = get_llm().with_structured_output(FeatureEngineeringPlan)

    async def plan(
        self,
        user_query: str,
        dataset_summary,
        eda_report: dict,
    ) -> FeatureEngineeringPlan:
        user_content = f"""User Query: {user_query}

Dataset Summary: {dataset_summary}

EDA Report: {eda_report}

Generate the feature engineering plan now."""

        messages = [
            {"role": "system", "content": FEATURE_ENGINEERING_PLANNER_PROMPT},
            {"role": "user", "content": user_content},
        ]

        response = await self.llm.ainvoke(messages)
        return response
