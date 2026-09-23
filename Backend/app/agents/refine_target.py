import logging
from llm.provider import get_llm
from schema.refine import RefineTarget
from prompts.refine_prompt import REFINE_TARGET_PROMPT
from utilis.llm_plan import invoke_with_repair

logger = logging.getLogger(__name__)


class RefineTargetAgent:
    async def identify(self, user_query: str, llm_config: dict | None = None) -> RefineTarget:
        logger.info("Identifying refinement target")

        llm = get_llm(llm_config).with_structured_output(RefineTarget)

        messages = [
            {"role": "system", "content": REFINE_TARGET_PROMPT},
            {"role": "user", "content": f"User request: {user_query}"},
        ]

        response = await invoke_with_repair(llm, messages)
        logger.info(
            f"Refine target identified: {response.target_stage} "
            f"(confidence={response.confidence})"
        )
        return response


refine_target_agent = RefineTargetAgent()
