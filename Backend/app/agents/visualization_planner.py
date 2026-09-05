import logging
from schema.visualization_plan import VisualizationPlanResponse
from prompts.visualization_prompt import visualization_prompt
from llm.provider import get_llm
from utilis.llm_plan import invoke_with_repair

logger = logging.getLogger(__name__)


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
        logger.info("Starting visualization planning")

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

        response = await invoke_with_repair(self.llm, messages)

        logger.info(
            f"Visualization plan generated: {len(response.visualizations)} charts")

        return response


visualization_planner_agent = VisualizationPlannerAgent()
