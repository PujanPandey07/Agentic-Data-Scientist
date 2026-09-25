# api/api_keys.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.db import get_db_session
from core.models import UserAPIKey
from core.security import get_current_user_id
from core.crypto import encrypt_api_key, decrypt_api_key
from schema.api_key import (
    APIKeySubmit, ModelChange, APIKeyStatus, ModelPreviewRequest, AVAILABLE_MODELS,
)
from services.live_models import fetch_live_models, LiveModelLookupError

router = APIRouter(prefix="/api/api-keys", tags=["API Keys"])


async def _get_own_key_row(session: AsyncSession, user_id: int) -> UserAPIKey | None:
    result = await session.execute(
        select(UserAPIKey).where(UserAPIKey.user_id == user_id)
    )
    return result.scalar_one_or_none()


@router.get("/available-models")
async def available_models():
    """Static placeholder list — no auth/key needed, just seeds the
    frontend dropdown before the user has typed a real key. Once a key
    exists, /preview-models (and PUT/PATCH validation) use the LIVE list
    instead."""
    return AVAILABLE_MODELS


@router.post("/preview-models")
async def preview_models(payload: ModelPreviewRequest):
    """Called by the frontend right after the user picks a provider and
    types a key — returns the models that key can actually use, live from
    the provider. Also doubles as key validation before the user commits
    to submitting it."""
    try:
        models = await fetch_live_models(payload.provider, payload.api_key)
    except LiveModelLookupError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"models": models}


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
    resubmitted.

    Validates model_name against the LIVE model list for this key, not the
    static AVAILABLE_MODELS — a stale model (e.g. a since-deprecated
    Gemini snapshot) is rejected here instead of failing later mid-pipeline."""
    try:
        live_models = await fetch_live_models(payload.provider, payload.api_key)
    except LiveModelLookupError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if payload.model_name not in live_models:
        raise HTTPException(
            status_code=400,
            detail=f"'{payload.model_name}' is not currently available for '{payload.provider}' "
                   f"with this key. Choose from: {live_models}",
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
    resubmission needed from the user, but the stored key IS decrypted
    server-side to re-validate the new model live against the provider,
    same as PUT does."""
    row = await _get_own_key_row(session, user_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail="No API key configured yet — submit one via PUT /api/api-keys first.",
        )

    decrypted_key = decrypt_api_key(row.encrypted_key)

    try:
        live_models = await fetch_live_models(row.provider, decrypted_key)
    except LiveModelLookupError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if payload.model_name not in live_models:
        raise HTTPException(
            status_code=400,
            detail=f"'{payload.model_name}' is not currently available for '{row.provider}' "
                   f"with this key. Choose from: {live_models}",
        )

    row.model_name = payload.model_name
    await session.commit()
    return APIKeyStatus(configured=True, provider=row.provider, model_name=row.model_name)


@router.get("/live-models")
async def get_live_models(
    user_id: int = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
):
    """Live model list for the CURRENTLY configured provider, using the
    already-stored key — no key resubmission needed. Powers the 'switch
    model' dropdown so it never shows a deprecated model."""
    row = await _get_own_key_row(session, user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No API key configured.")

    decrypted_key = decrypt_api_key(row.encrypted_key)
    try:
        models = await fetch_live_models(row.provider, decrypted_key)
    except LiveModelLookupError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"provider": row.provider, "models": models}


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
