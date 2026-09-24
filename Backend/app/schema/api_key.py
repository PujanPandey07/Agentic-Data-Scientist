# schema/api_keys.py
from pydantic import BaseModel, Field
from typing import Literal

Provider = Literal["openai", "anthropic", "gemini", "groq"]

# Hardcoded and manually maintained rather than fetched live from each
# provider's API — avoids trusting a live response to shape the frontend
# dropdown, and keeps deprecated/irrelevant model types (embeddings, etc.)
# out of the list entirely. Update this list by hand as new models ship.
AVAILABLE_MODELS: dict[str, list[str]] = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    "anthropic": ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest", "claude-3-opus-latest"],
    "gemini": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
    "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
}


class APIKeySubmit(BaseModel):
    """Submitting a NEW provider+key — always required together, since
    switching provider means the old key is no longer relevant at all."""
    provider: Provider
    api_key: str = Field(min_length=1)
    model_name: str


class ModelChange(BaseModel):
    """Switching model WITHIN the currently configured provider — no key
    needed, since the key doesn't change."""
    model_name: str


class APIKeyStatus(BaseModel):
    """What we return to the frontend — provider + model only. The key
    itself is write-only and never sent back, by design."""
    configured: bool
    provider: Provider | None = None
    model_name: str | None = None
