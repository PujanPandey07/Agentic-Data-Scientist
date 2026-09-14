# api/conversations.py — full updated file
from fastapi import APIRouter, Depends
from sqlalchemy import select
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
