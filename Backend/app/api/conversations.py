# api/conversations.py — full updated file
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, delete, text
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
from services.short_term_memory import short_term_memory_manager

logger = logging.getLogger(__name__)

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

    Also cleans up:
    - short_term_memories row keyed by conversation_id / thread_id
    - LangGraph checkpoint rows (checkpoints, checkpoint_blobs,
      checkpoint_writes) keyed by thread_id
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

    # ── Clean up short-term memory row for this conversation ────────────────
    try:
        await short_term_memory_manager.invalidate(thread_id)
    except Exception:
        logger.warning(
            "Could not invalidate short-term memory for thread %s", thread_id,
            exc_info=True,
        )

    # ── Clean up LangGraph checkpoint rows keyed by thread_id ───────────────
    # LangGraph stores checkpoints in three tables with a thread_id text column.
    # We issue raw DELETE statements because LangGraph doesn't expose a cleanup
    # API we can call safely here.
    _LANGGRAPH_TABLES = (
        "checkpoint_blobs",
        "checkpoint_writes",
        "checkpoints",
    )
    for table in _LANGGRAPH_TABLES:
        try:
            await session.execute(
                text(f"DELETE FROM {table} WHERE thread_id = :tid"),  # noqa: S608
                {"tid": thread_id},
            )
        except Exception:
            # Table may not exist in all deployment configurations — log and
            # continue so the delete endpoint never fails because of this.
            logger.warning(
                "Could not delete LangGraph rows from %s for thread %s",
                table, thread_id, exc_info=True,
            )
    try:
        await session.commit()
    except Exception:
        logger.warning(
            "Could not commit LangGraph checkpoint cleanup for thread %s",
            thread_id, exc_info=True,
        )
