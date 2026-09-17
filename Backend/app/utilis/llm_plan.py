import asyncio
import logging
import re

logger = logging.getLogger(__name__)

RETRY_AFTER_PATTERN = re.compile(r"try again in\s+([\d.]+)s", re.IGNORECASE)


def _retry_after_seconds(error: Exception) -> float | None:
    match = RETRY_AFTER_PATTERN.search(str(error))
    return float(match.group(1)) if match else None


async def invoke_with_repair(llm, messages: list[dict]):
    """
    Call a structured-output LLM. If it fails to produce valid output,
    retry once with the error fed back so the model can self-correct.
    """
    try:
        return await llm.ainvoke(messages)
    except Exception as first_error:
        error_text = str(first_error).lower()
        if "request too large" in error_text:
            logger.error(
                "LLM request exceeded the provider token limit; skipping repair retry: %s",
                first_error,
            )
            raise

        retry_after = _retry_after_seconds(first_error)
        if "rate_limit" in error_text or "rate limit" in error_text:
            wait_seconds = min(
                retry_after if retry_after is not None else 3.0, 15.0)
            logger.warning(
                "LLM rate limit reached; waiting %.1fs before one retry.",
                wait_seconds,
            )
            await asyncio.sleep(wait_seconds)
            try:
                return await llm.ainvoke(messages)
            except Exception as retry_error:
                logger.error("LLM rate-limit retry failed: %s", retry_error)
                raise

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
