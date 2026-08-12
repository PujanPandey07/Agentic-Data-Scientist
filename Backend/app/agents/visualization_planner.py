from schema.visualization_plan import VisualizationPlanResponse
from prompts.visualization_prompt import visualization_prompt
from llm.provider import get_llm


class VisualizationPlannerAgent:

    def __init__(self):
        self.llm = get_llm().with_structured_output(
            VisualizationPlanResponse
        )

    async def plan_visualizations(
        self,
        user_query: str,
        dataset_summary,
        eda_report: dict,
    ) -> VisualizationPlanResponse:

        messages = [
            {
                "role": "system",
                "content": visualization_prompt,
            },
            {
                "role": "user",
                "content": f"""
User Query:
{user_query}

Dataset Summary:
{dataset_summary}

EDA Report:
{eda_report}
""",
            },
        ]

        response = await self.llm.ainvoke(messages)

        return response


visualization_planner_agent = VisualizationPlannerAgent()
