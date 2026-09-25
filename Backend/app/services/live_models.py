"""Live 'what models can this key actually use' lookups, one per provider.
Replaces trusting a hand-maintained static list at the point a key is
actually being submitted or a model actually being switched to — a dead
model (see the gemini-2.0-flash 404 in prod) gets caught immediately
instead of failing mid-pipeline later.

Each function takes a plaintext API key and returns a list of bare model-id
strings in the SAME format schema/api_keys.AVAILABLE_MODELS already uses
(e.g. "gpt-4o", not "models/gpt-4o"), raises LiveModelLookupError on any
failure (bad key, network error, unexpected response shape) with a message
safe to surface to the user.
"""
import httpx


class LiveModelLookupError(Exception):
    pass


_TIMEOUT = 15.0


async def fetch_openai_models(api_key: str) -> list[str]:
    url = "https://api.openai.com/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url, headers=headers)
    except httpx.RequestError as e:
        raise LiveModelLookupError(f"Could not reach OpenAI: {e}") from e

    if resp.status_code == 401:
        raise LiveModelLookupError("OpenAI rejected this API key.")
    if resp.status_code != 200:
        raise LiveModelLookupError(
            f"OpenAI returned {resp.status_code}: {resp.text[:200]}")

    data = resp.json().get("data", [])
    excluded_markers = (
        "embedding", "audio", "realtime", "transcribe", "tts",
        "whisper", "moderation", "dall-e", "instruct",
    )
    models = [
        m["id"] for m in data
        if (m["id"].startswith("gpt-") or m["id"].startswith("o1") or m["id"].startswith("o3"))
        and not any(marker in m["id"] for marker in excluded_markers)
    ]
    if not models:
        raise LiveModelLookupError(
            "OpenAI key is valid but returned no usable chat models.")
    return sorted(set(models))


async def fetch_anthropic_models(api_key: str) -> list[str]:
    url = "https://api.anthropic.com/v1/models"
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url, headers=headers)
    except httpx.RequestError as e:
        raise LiveModelLookupError(f"Could not reach Anthropic: {e}") from e

    if resp.status_code == 401:
        raise LiveModelLookupError("Anthropic rejected this API key.")
    if resp.status_code != 200:
        raise LiveModelLookupError(
            f"Anthropic returned {resp.status_code}: {resp.text[:200]}")

    data = resp.json().get("data", [])
    # Note: Anthropic's /v1/models lists concrete dated snapshots
    # (e.g. "claude-3-5-sonnet-20241022"), not the "-latest" aliases the
    # old static list used — aliases aren't enumerable via this endpoint.
    models = [m["id"] for m in data if "claude" in m["id"]]
    if not models:
        raise LiveModelLookupError(
            "Anthropic key is valid but returned no usable models.")
    return sorted(set(models))


async def fetch_gemini_models(api_key: str) -> list[str]:
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url, params={"key": api_key})
    except httpx.RequestError as e:
        raise LiveModelLookupError(f"Could not reach Gemini: {e}") from e

    if resp.status_code in (400, 401, 403):
        raise LiveModelLookupError("Gemini rejected this API key.")
    if resp.status_code != 200:
        raise LiveModelLookupError(
            f"Gemini returned {resp.status_code}: {resp.text[:200]}")

    data = resp.json().get("models", [])
    models = [
        m["name"].removeprefix("models/")
        for m in data
        if "generateContent" in m.get("supportedGenerationMethods", [])
    ]
    if not models:
        raise LiveModelLookupError(
            "Gemini key is valid but returned no usable chat models.")
    return sorted(set(models))


async def fetch_groq_models(api_key: str) -> list[str]:
    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url, headers=headers)
    except httpx.RequestError as e:
        raise LiveModelLookupError(f"Could not reach Groq: {e}") from e

    if resp.status_code == 401:
        raise LiveModelLookupError("Groq rejected this API key.")
    if resp.status_code != 200:
        raise LiveModelLookupError(
            f"Groq returned {resp.status_code}: {resp.text[:200]}")

    data = resp.json().get("data", [])
    excluded_markers = ("whisper", "tts", "guard")
    models = [m["id"] for m in data if not any(
        marker in m["id"] for marker in excluded_markers)]
    if not models:
        raise LiveModelLookupError(
            "Groq key is valid but returned no usable chat models.")
    return sorted(set(models))


FETCHERS = {
    "openai": fetch_openai_models,
    "anthropic": fetch_anthropic_models,
    "gemini": fetch_gemini_models,
    "groq": fetch_groq_models,
}


async def fetch_live_models(provider: str, api_key: str) -> list[str]:
    fetcher = FETCHERS.get(provider)
    if fetcher is None:
        raise LiveModelLookupError(f"Unknown provider '{provider}'.")
    return await fetcher(api_key)
