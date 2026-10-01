import asyncio
import logging
import re

logger = logging.getLogger(__name__)

RETRY_AFTER_PATTERN = re.compile(r"try again in\s+([\d.]+)s", re.IGNORECASE)


class LLMUserError(Exception):
    """Raised when an LLM call fails due to a user-actionable cause
    (bad API key, exhausted quota, context too large). The message is
    safe to display directly in the chat UI."""


def _classify_llm_error(error: Exception) -> str | None:
    """
    Return a user-facing message if the error is a known user-actionable
    LLM failure, or None if it's a transient/unknown error that should
    follow the normal retry path.
    """
    text = str(error).lower()

    # Auth / invalid key — user needs to check their BYOK settings
    if any(k in text for k in (
        "invalid api key", "incorrect api key", "authentication",
        "unauthorized", "invalid_api_key", "permission_denied",
        "api key not valid", "invalid key",
    )):
        return (
            "Your API key was rejected by the provider. "
            "Please go to Settings → API Keys, check that your key is correct and still active, "
            "then try again."
        )

    # Quota / billing exhausted — user's credits are gone
    if any(k in text for k in (
        "insufficient_quota", "insufficient quota", "quota exceeded",
        "billing", "you exceeded your current quota",
        "exceeded your monthly", "rate limit exceeded", "you have run out",
        "account has been deactivated", "no credits",
    )):
        return (
            "Your API key has run out of credits or hit its usage limit. "
            "Please top up your account with the provider, or go to Settings → API Keys "
            "to remove your key and fall back to the shared system model."
        )

    # Context too large — dataset summary or prompt is too big
    if any(k in text for k in (
        "request too large", "context_length_exceeded", "context length",
        "maximum context length", "token limit", "too many tokens",
        "string too long",
    )):
        return (
            "The request was too large for the model's context window. "
            "Try uploading a smaller dataset, or use a model with a larger context limit."
        )

    return None


def _retry_after_seconds(error: Exception) -> float | None:
    match = RETRY_AFTER_PATTERN.search(str(error))
    return float(match.group(1)) if match else None


async def invoke_with_repair(llm, messages: list[dict]):
    """
    Call a structured-output LLM. If it fails to produce valid output,
    retry once with the error fed back so the model can self-correct.
    Raises LLMUserError immediately (no retry) for auth/quota/context failures.
    """
    try:
        return await llm.ainvoke(messages)
    except Exception as first_error:
        # Check for user-actionable errors first — no point retrying these
        user_msg = _classify_llm_error(first_error)
        if user_msg:
            logger.error("LLM user-actionable error: %s", first_error)
            raise LLMUserError(user_msg) from first_error

        error_text = str(first_error).lower()

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
                user_msg = _classify_llm_error(retry_error)
                if user_msg:
                    raise LLMUserError(user_msg) from retry_error
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
            user_msg = _classify_llm_error(second_error)
            if user_msg:
                raise LLMUserError(user_msg) from second_error
            logger.error(f"LLM call failed again after retry: {second_error}")
            raise

