# schema/api_keys.py
from pydantic import BaseModel, Field
from typing import Literal

Provider = Literal["openai", "anthropic", "gemini"]

# Hardcoded and manually maintained rather than fetched live from each
# provider's API — avoids trusting a live response to shape the frontend
# dropdown, and keeps deprecated/irrelevant model types (embeddings, etc.)
# out of the list entirely. Update this list by hand as new models ship.
AVAILABLE_MODELS: dict[str, list[str]] = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4.1"],
    "anthropic": ["claude-sonnet-4-5", "claude-opus-4-1", "claude-haiku-4-5"],
    "gemini": ["gemini-2.5-flash", "gemini-2.5-pro"],
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
