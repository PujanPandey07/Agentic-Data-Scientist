# agents/model_selection_planner.py
import logging
from llm.provider import get_llm
from schema.model_selection import ModelSelectionPlan
from prompts.model_selection import MODEL_SELECTION_PLANNER_PROMPT
from utilis.llm_plan import invoke_with_repair
from utilis.constraints import format_constraints
from utilis.prompt_context import summarize_eda_report_for_prompt, truncate_for_prompt

logger = logging.getLogger(__name__)


class ModelSelectionPlannerAgent:
    # No self.llm in __init__ — see planner.py for why: a module-level
    # singleton can't hold a per-user LLM client, since it's built once at
    # import time before any user's llm_config exists.

    async def plan(
        self,
        user_query: str,
        dataset_summary,
        eda_report: dict,
        feature_engineering_report: dict | None,
        constraints: list[str] | None = None,
        llm_config: dict | None = None,
    ) -> ModelSelectionPlan:
        logger.info("Starting model selection planning")

        llm = get_llm(llm_config).with_structured_output(ModelSelectionPlan)

        fe_section = ""
        if feature_engineering_report:
            fe_section = f"\nFeature Engineering Report: {truncate_for_prompt(feature_engineering_report)}"

        user_content = f"""User Query: {user_query}

Dataset Summary: {truncate_for_prompt(dataset_summary)}

EDA Report: {summarize_eda_report_for_prompt(eda_report)}{fe_section}{format_constraints(constraints)}

Generate the model selection plan now."""

        messages = [
            {"role": "system", "content": MODEL_SELECTION_PLANNER_PROMPT},
            {"role": "user", "content": user_content},
        ]

        response = await invoke_with_repair(llm, messages)

        logger.info(
            f"Model selection plan generated: {len(response.candidates)} candidates")

        return response


model_selection_planner_agent = ModelSelectionPlannerAgent()
