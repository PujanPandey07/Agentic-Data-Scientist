import logging
from llm.provider import get_llm
from schema.intent import IntentClassification
from prompts.intent_prompt import INTENT_ROUTER_PROMPT
from utilis.llm_plan import invoke_with_repair

logger = logging.getLogger(__name__)


class IntentRouterAgent:
    def __init__(self):
        self.llm = get_llm().with_structured_output(IntentClassification)

    async def classify(self, user_query: str, has_prior_report: bool) -> IntentClassification:
        logger.info("Classifying user intent")

        messages = [
            {"role": "system", "content": INTENT_ROUTER_PROMPT},
            {"role": "user", "content": f"""User message: {user_query}

A previous analysis report exists in this session: {has_prior_report}

Classify the intent now."""},
        ]

        response = await invoke_with_repair(self.llm, messages)
        logger.info(
            f"Intent classified: {response.intent} — {response.reasoning}")
        return response


intent_router_agent = IntentRouterAgent()
