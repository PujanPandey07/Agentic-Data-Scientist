import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Try to import Groq, but don't crash if not installed
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

load_dotenv()


def get_llm():
    """Get LLM based on PROVIDER env variable. Defaults to gemini."""
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    if provider == "groq":
        if not GROQ_AVAILABLE:
            raise ImportError(
                "Groq support not installed. Run: pip install langchain-groq"
            )

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        model = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

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
