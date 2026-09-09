import logging
from llm.provider import get_llm
from schema.model_selection import ModelSelectionPlan
from prompts.model_selection import MODEL_SELECTION_PLANNER_PROMPT
from utilis.llm_plan import invoke_with_repair
from utilis.constraints import format_constraints

logger = logging.getLogger(__name__)


class ModelSelectionPlannerAgent:
    def __init__(self):
        self.llm = get_llm().with_structured_output(ModelSelectionPlan)

    async def plan(
        self,
        user_query: str,
        dataset_summary,
        eda_report: dict,
        feature_engineering_report: dict | None,
        constraints: list[str] | None = None,
    ) -> ModelSelectionPlan:
        logger.info("Starting model selection planning")

        fe_section = ""
        if feature_engineering_report:
            fe_section = f"\nFeature Engineering Report: {feature_engineering_report}"

        user_content = f"""User Query: {user_query}

Dataset Summary: {dataset_summary}

EDA Report: {eda_report}{fe_section}{format_constraints(constraints)}

Generate the model selection plan now."""

        messages = [
            {"role": "system", "content": MODEL_SELECTION_PLANNER_PROMPT},
            {"role": "user", "content": user_content},
        ]

        response = await invoke_with_repair(self.llm, messages)

        logger.info(
            f"Model selection plan generated: {len(response.candidates)} candidates")

        return response
