import logging
from llm.provider import get_llm
from schema.constraints import UserConstraints
from prompts.constraint_prompt import CONSTRAINTS_EXTRACTION_PROMPT
from utilis.llm_plan import invoke_with_repair

logger = logging.getLogger(__name__)


class ConstraintsExtractorAgent:
    def __init__(self):
        self.llm = get_llm().with_structured_output(UserConstraints)

    async def extract(self, user_query: str) -> UserConstraints:
        logger.info("Extracting user constraints")

        messages = [
            {"role": "system", "content": CONSTRAINTS_EXTRACTION_PROMPT},
            {"role": "user", "content": f"User request: {user_query}"},
        ]

        response = await invoke_with_repair(self.llm, messages)
        logger.info(
            f"Constraints extracted: {len(response.constraints)} found")
        return response


constraints_extractor_agent = ConstraintsExtractorAgent()
