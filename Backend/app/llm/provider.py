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


def get_llm(user_llm_config: dict | None = None):
    """Get an LLM client.

    If user_llm_config is provided (shape: {"provider", "api_key",
    "model_name"} — see resolve_user_llm_config below), builds a client
    using THAT user's own provider + key + model choice. Otherwise falls
    back to the original env-based default (LLM_PROVIDER/.env) — this
    keeps every existing call site that doesn't pass a config working
    exactly as before.
    """
    if user_llm_config is not None:
        return _build_client(
            provider=user_llm_config["provider"],
            api_key=user_llm_config["api_key"],
            model_name=user_llm_config["model_name"],
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

        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

        return ChatGroq(
            model=model,
            temperature=0,
            api_key=api_key,
            max_retries=2,
        )

    # Default: Gemini
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
    )


def _build_client(provider: str, api_key: str, model_name: str):
    """Constructs the right LangChain client for a user-supplied
    provider+key+model. Kept separate from get_llm()'s env-fallback logic
    so the two paths (shared default vs. user's own key) stay clearly
    distinct and don't accidentally cross-contaminate."""
    if provider == "openai":
        return ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=api_key,
            max_retries=2,
        )
    elif provider == "anthropic":
        return ChatAnthropic(
            model=model_name,
            temperature=0,
            api_key=api_key,
            max_retries=2,
        )
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            google_api_key=api_key,
        )

    raise ValueError(f"Unsupported provider: {provider}")
