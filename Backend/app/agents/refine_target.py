import logging
from llm.provider import get_llm
from schema.refine import RefineTarget
from prompts.refine_prompt import REFINE_TARGET_PROMPT
from utilis.llm_plan import invoke_with_repair

logger = logging.getLogger(__name__)


class RefineTargetAgent:
    def __init__(self):
        self.llm = get_llm().with_structured_output(RefineTarget)

    async def identify(self, user_query: str) -> RefineTarget:
        logger.info("Identifying refinement target")

        messages = [
            {"role": "system", "content": REFINE_TARGET_PROMPT},
            {"role": "user", "content": f"User request: {user_query}"},
        ]

        response = await invoke_with_repair(self.llm, messages)
        logger.info(
            f"Refine target identified: {response.target_stage} "
            f"(confidence={response.confidence})"
        )
        return response


refine_target_agent = RefineTargetAgent()
