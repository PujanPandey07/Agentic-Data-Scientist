# api/conversations.py — full updated file
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from core.db import get_db_session
from core.models import Conversation, Message
from core.security import get_current_user_id
from schema.conversation import (
    ConversationSummary,
    ConversationListResponse,
    MessageInfo,
    MessageListResponse,
)
from api.chat import _get_owned_conversation

router = APIRouter(prefix="/api/conversations", tags=["Conversations"])


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    user_id: int = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
):
    result = await session.execute(
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.created_at.desc())
    )
    conversations = result.scalars().all()

    return ConversationListResponse(
        conversations=[
            ConversationSummary(
                thread_id=c.thread_id,
                dataset_id=c.dataset_id,
                title=c.title,
                created_at=c.created_at,
            )
            for c in conversations
        ]
    )


@router.get("/{thread_id}/messages", response_model=MessageListResponse)
async def get_messages(
    thread_id: str,
    user_id: int = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
):
    conversation = await _get_owned_conversation(session, thread_id, user_id)

    result = await session.execute(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at.asc())
    )
    messages = result.scalars().all()

    return MessageListResponse(
        thread_id=thread_id,
        messages=[
            MessageInfo(role=m.role, content=m.content,
                        created_at=m.created_at)
            for m in messages
        ],
    )


@router.delete("/{thread_id}", status_code=204)
async def delete_conversation(
    thread_id: str,
    user_id: int = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
):
    """Delete a conversation and all its messages.
    Returns 204 No Content on success — same deliberate 404 for
    'not found' and 'belongs to someone else' to avoid leaking
    whether a thread_id exists at all.
    """
    # Ownership check — raises 404 if not found or not theirs.
    conversation = await _get_owned_conversation(session, thread_id, user_id)

    # Delete child messages first to satisfy the foreign-key constraint,
    # then delete the conversation row itself.
    await session.execute(
        delete(Message).where(Message.conversation_id == conversation.id)
    )
    await session.delete(conversation)
    await session.commit()
