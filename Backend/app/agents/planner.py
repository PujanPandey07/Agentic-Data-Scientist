from langchain_core.messages import HumanMessage, SystemMessage

from llm.provider import get_llm
from prompts.planner_prompt import PLANNER_SYSTEM_PROMPT
from schema.analysis_plan import AnalysisPlan


class PlannerAgent:

    def __init__(self):
        self.llm = get_llm().with_structured_output(AnalysisPlan)

    async def plan(self, user_query, dataset_summary):

        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(
                content=f"""
User Goal:
{user_query}

Dataset Summary:
{dataset_summary.model_dump_json(indent=2)}
"""
            ),
        ]

        response = await self.llm.ainvoke(messages)

        return response


planner_agent = PlannerAgent()
