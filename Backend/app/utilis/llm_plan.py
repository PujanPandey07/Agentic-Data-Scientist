# app/utils/llm_plan.py
import logging

logger = logging.getLogger(__name__)


async def invoke_with_repair(llm, messages: list[dict]):
    """
    Call a structured-output LLM. If it fails to produce valid output,
    retry once with the error fed back so the model can self-correct.
    """
    try:
        return await llm.ainvoke(messages)
    except Exception as first_error:
        logger.warning(
            f"LLM call failed, retrying with error feedback: {first_error}")

        repair_message = {
            "role": "user",
            "content": (
                f"Your previous response could not be parsed. "
                f"Error: {first_error}\n\n"
                f"Please try again and return a valid response that strictly "
                f"matches the required format."
            ),
        }

        try:
            return await llm.ainvoke(messages + [repair_message])
        except Exception as second_error:
            logger.error(f"LLM call failed again after retry: {second_error}")
            raise
