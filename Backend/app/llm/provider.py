# llm/provider.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Try to import Groq, but don't crash if not installed
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

load_dotenv()


import logging

logger = logging.getLogger(__name__)

# Normalize known user typos / unreleased version names to canonical model IDs
MODEL_NAME_ALIASES = {
    "gemini-2.5-flash": "gemini-2.0-flash",
    "gemini-2.5-pro": "gemini-1.5-pro",
    "gpt-4.1": "gpt-4o",
    "claude-sonnet-4-5": "claude-3-5-sonnet-latest",
    "claude-opus-4-1": "claude-3-opus-latest",
    "claude-haiku-4-5": "claude-3-5-haiku-latest",
}


def get_llm(user_llm_config: dict | None = None):
    """Get an LLM client.

    If user_llm_config is provided (shape: {"provider", "api_key",
    "model_name"}), builds a client using that configuration.
    Falls back to the environment-based default if missing or invalid.
    """
    if user_llm_config is not None:
        try:
            return _build_client(
                provider=user_llm_config.get("provider", "gemini"),
                api_key=user_llm_config.get("api_key", ""),
                model_name=user_llm_config.get("model_name", "gemini-2.0-flash"),
            )
        except Exception as e:
            logger.warning(
                f"Failed to build user LLM client: {e}. Falling back to default system LLM."
            )

    provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    if provider == "groq":
        if not GROQ_AVAILABLE:
            raise ImportError(
                "Groq support not installed. Run: pip install langchain-groq"
            )

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

        return ChatGroq(
            model=model,
            temperature=0,
            api_key=api_key,
            max_retries=2,
        )

    # Default: Gemini (using official released model identifier)
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
    )


def _build_client(provider: str, api_key: str, model_name: str):
    """Constructs the right LangChain client for a user-supplied provider+key+model."""
    canonical_model = MODEL_NAME_ALIASES.get(model_name, model_name)

    if provider == "openai":
        return ChatOpenAI(
            model=canonical_model,
            temperature=0,
            api_key=api_key,
            max_retries=2,
        )
    elif provider == "anthropic":
        return ChatAnthropic(
            model=canonical_model,
            temperature=0,
            api_key=api_key,
            max_retries=2,
        )
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=canonical_model,
            temperature=0,
            google_api_key=api_key,
        )
    elif provider == "groq":
        if not GROQ_AVAILABLE:
            raise ImportError(
                "Groq support not installed. Run: pip install langchain-groq"
            )
        return ChatGroq(
            model=canonical_model,
            temperature=0,
            api_key=api_key,
            max_retries=2,
        )

    raise ValueError(f"Unsupported provider: {provider}")
