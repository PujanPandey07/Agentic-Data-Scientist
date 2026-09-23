# services/llm_config.py
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.models import UserAPIKey
from core.crypto import decrypt_api_key


async def resolve_user_llm_config(session: AsyncSession, user_id: int | None) -> dict | None:
    """Looks up the user's stored provider+key+model, decrypts the key, and
    returns it in the shape get_llm() expects. Returns None if the user
    has no key configured (or user_id itself is None) — callers should
    treat None as "use the shared default", not as an error.
    """
    if user_id is None:
        return None

    result = await session.execute(
        select(UserAPIKey).where(UserAPIKey.user_id == user_id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        return None

    return {
        "provider": row.provider,
        "api_key": decrypt_api_key(row.encrypted_key),
        "model_name": row.model_name,
    }
