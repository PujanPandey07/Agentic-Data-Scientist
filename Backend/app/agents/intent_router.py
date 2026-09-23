import logging
from llm.provider import get_llm
from schema.intent import IntentClassification
from prompts.intent_prompt import INTENT_ROUTER_PROMPT
from utilis.llm_plan import invoke_with_repair

logger = logging.getLogger(__name__)


class IntentRouterAgent:
    async def classify(
        self,
        user_query: str,
        has_prior_report: bool,
        llm_config: dict | None = None,
    ) -> IntentClassification:
        logger.info("Classifying user intent")

        llm = get_llm(llm_config).with_structured_output(IntentClassification)

        messages = [
            {"role": "system", "content": INTENT_ROUTER_PROMPT},
            {"role": "user", "content": f"""User message: {user_query}

A previous analysis report exists in this session: {has_prior_report}

Classify the intent now."""},
        ]

        response = await invoke_with_repair(llm, messages)
        logger.info(
            f"Intent classified: {response.intent} — {response.reasoning}")
        return response


intent_router_agent = IntentRouterAgent()
