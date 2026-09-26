import logging

from llm.provider import get_llm
from prompts.clusterinmg import CLUSTERING_MODEL_SELECTION_PROMPT
from schema.clustering import ClusteringPlan
from utilis.llm_plan import invoke_with_repair

logger = logging.getLogger(__name__)


class ClusteringModelSelectionPlannerAgent:
    """LLM reasoning: decides WHICH clustering algorithms to try. Mirrors
    ModelSelectionPlannerAgent's shape (no self.llm built at import time —
    a fresh client is built per call so per-user BYOK configs work)."""

    async def plan(
        self,
        user_query: str,
        dataset_summary,
        eda_report: dict,
        feature_engineering_report: dict | None,
        constraints: list[str],
        llm_config: dict | None = None,
    ) -> ClusteringPlan:
        logger.info("Starting clustering model selection planning")

        llm = get_llm(llm_config).with_structured_output(ClusteringPlan)

        constraints_text = (
            "\n".join(f"- {c}" for c in constraints) if constraints else "None"
        )

        messages = [
            {"role": "system", "content": CLUSTERING_MODEL_SELECTION_PROMPT},
            {"role": "user", "content": f"""
User query:
{user_query}

Dataset summary:
{dataset_summary.model_dump_json(indent=2) if dataset_summary else "Not available"}

EDA report:
{eda_report}

Feature engineering report:
{feature_engineering_report or "Not available"}

User constraints for this stage:
{constraints_text}

Produce the ClusteringPlan now.
"""},
        ]

        response = await invoke_with_repair(llm, messages)

        logger.info(
            f"Clustering plan generated: strategy={response.strategy}, "
            f"{len(response.candidates)} candidates, metric={response.scoring_metric}"
        )

        return response


clustering_model_selection_planner_agent = ClusteringModelSelectionPlannerAgent()
