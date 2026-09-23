import logging
from langchain_core.messages import HumanMessage, SystemMessage

from llm.provider import get_llm
from prompts.planner_prompt import PLANNER_SYSTEM_PROMPT
from schema.analysis_plan import AnalysisPlan
from utilis.llm_plan import invoke_with_repair

logger = logging.getLogger(__name__)


class PlannerAgent:
    # No self.llm built in __init__ anymore — a module-level singleton
    # instance can't hold a per-user LLM client, since it's constructed
    # once at import time, before any request (or any user) exists.

    async def plan(self, user_query, dataset_summary, llm_config: dict | None = None):
        logger.info("Starting main analysis planning")

        llm = get_llm(llm_config).with_structured_output(AnalysisPlan)

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

        response = await invoke_with_repair(llm, messages)

        logger.info(
            f"Analysis plan generated: problem_type={response.problem_type}, {len(response.tasks)} tasks")

        return response


planner_agent = PlannerAgent()
