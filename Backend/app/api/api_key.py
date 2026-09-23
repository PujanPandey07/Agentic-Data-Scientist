# api/api_keys.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.db import get_db_session
from core.models import UserAPIKey
from core.security import get_current_user_id
from core.crypto import encrypt_api_key
from schema.api_key import APIKeySubmit, ModelChange, APIKeyStatus, AVAILABLE_MODELS

router = APIRouter(prefix="/api/api-keys", tags=["API Keys"])


async def _get_own_key_row(session: AsyncSession, user_id: int) -> UserAPIKey | None:
    result = await session.execute(
        select(UserAPIKey).where(UserAPIKey.user_id == user_id)
    )
    return result.scalar_one_or_none()


@router.get("/available-models")
async def available_models():
    """Static curated list — no auth needed, just populates the frontend
    dropdown before/after a provider is chosen."""
    return AVAILABLE_MODELS


@router.get("", response_model=APIKeyStatus)
async def get_status(
    user_id: int = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
):
    row = await _get_own_key_row(session, user_id)
    if row is None:
        return APIKeyStatus(configured=False)
    return APIKeyStatus(configured=True, provider=row.provider, model_name=row.model_name)


@router.put("", response_model=APIKeyStatus)
async def submit_key(
    payload: APIKeySubmit,
    user_id: int = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
):
    """Create or fully replace the user's single active configuration.
    Always requires a fresh api_key — even if they're only changing the
    model, this endpoint assumes a provider switch (or a genuine key
    rotation) and re-encrypts from scratch. For model-only changes within
    the SAME provider, use PATCH /model instead so the key isn't needlessly
    resubmitted."""
    valid_models = AVAILABLE_MODELS.get(payload.provider, [])
    if payload.model_name not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"'{payload.model_name}' is not a supported model for '{payload.provider}'. "
                   f"Choose from: {valid_models}",
        )

    encrypted = encrypt_api_key(payload.api_key)
    row = await _get_own_key_row(session, user_id)

    if row is None:
        row = UserAPIKey(
            user_id=user_id,
            provider=payload.provider,
            encrypted_key=encrypted,
            model_name=payload.model_name,
        )
        session.add(row)
    else:
        row.provider = payload.provider
        row.encrypted_key = encrypted
        row.model_name = payload.model_name

    await session.commit()
    return APIKeyStatus(configured=True, provider=row.provider, model_name=row.model_name)


@router.patch("/model", response_model=APIKeyStatus)
async def change_model(
    payload: ModelChange,
    user_id: int = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
):
    """Switch model WITHIN the currently configured provider — no key
    resubmission needed."""
    row = await _get_own_key_row(session, user_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail="No API key configured yet — submit one via PUT /api/api-keys first.",
        )

    valid_models = AVAILABLE_MODELS.get(row.provider, [])
    if payload.model_name not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"'{payload.model_name}' is not a supported model for '{row.provider}'. "
                   f"Choose from: {valid_models}",
        )

    row.model_name = payload.model_name
    await session.commit()
    return APIKeyStatus(configured=True, provider=row.provider, model_name=row.model_name)


@router.delete("")
async def delete_key(
    user_id: int = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
):
    """Remove the user's configuration entirely — they fall back to the
    shared default provider (existing env-based get_llm() behavior)."""
    row = await _get_own_key_row(session, user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No API key configured.")

    await session.delete(row)
    await session.commit()
    return {"deleted": True}
