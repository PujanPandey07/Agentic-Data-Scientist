# api/chat.py — full updated file
from fastapi import APIRouter, Depends, HTTPException, Request
from langgraph.types import Command
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.db import get_db_session
from core.models import Conversation, Message
from core.security import get_current_user_id
from schema.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/api", tags=["Chat"])


async def _get_owned_conversation(session: AsyncSession, thread_id: str, user_id: int) -> Conversation:
    result = await session.execute(
        select(Conversation).where(
            Conversation.thread_id == thread_id,
            Conversation.user_id == user_id,
        )
    )
    conversation = result.scalar_one_or_none()
    if conversation is None:
        # Deliberately the same 404 whether the thread doesn't exist at
        # all OR belongs to someone else — don't reveal that a thread_id
        # exists but isn't theirs.
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    user_id: int = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
):
    conversation = await _get_owned_conversation(session, payload.thread_id, user_id)

    graph = request.app.state.graph
    config = {"configurable": {"thread_id": payload.thread_id}}

    snapshot = await graph.aget_state(config)
    is_paused = bool(snapshot.next) and any(
        task.interrupts for task in snapshot.tasks
    )

    if is_paused:
        if payload.decision is None:
            raise HTTPException(
                status_code=400,
                detail="This conversation is waiting for a decision (approve/reject/edit), not a new message.",
            )
        resume_payload = {
            "approved": payload.decision.approved,
            "edit_instruction": payload.decision.edit_instruction,
        }
        # Log the decision itself as the "user" turn, in readable form.
        user_message_content = (
            payload.decision.edit_instruction
            if payload.decision.edit_instruction
            else ("approved" if payload.decision.approved else "rejected")
        )
        result = await graph.ainvoke(Command(resume=resume_payload), config=config)
    else:
        if payload.user_query is None:
            raise HTTPException(
                status_code=400,
                detail="user_query is required when the conversation isn't waiting on a decision.",
            )
        user_message_content = payload.user_query
        follow_up_state = {
            "user_query": payload.user_query,
            "dataset_id": snapshot.values.get("dataset_id"),
        }
        result = await graph.ainvoke(follow_up_state, config=config)

    interrupted = "__interrupt__" in result
    interrupt_payload = result["__interrupt__"][0].value if interrupted else None

    # What actually gets logged as the "assistant" turn — prefer a direct
    # answer, fall back to the interrupt's summary, fall back to a
    # generic note so there's always SOMETHING recorded.
    if result.get("direct_answer"):
        assistant_message_content = result["direct_answer"]
    elif interrupted:
        assistant_message_content = interrupt_payload.get(
            "summary", "Waiting for your input.")
    else:
        assistant_message_content = "Pipeline step completed."

    session.add(Message(
        conversation_id=conversation.id,
        role="user",
        content=user_message_content,
    ))
    session.add(Message(
        conversation_id=conversation.id,
        role="assistant",
        content=assistant_message_content,
    ))
    await session.commit()

    return ChatResponse(
        interrupted=interrupted,
        interrupt=interrupt_payload,
        intent=result.get("intent"),
        direct_answer=result.get("direct_answer"),
    )
