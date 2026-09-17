# api/chat.py — full updated file
from fastapi import APIRouter, Depends, HTTPException, Request
from langgraph.types import Command
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.db import get_db_session
from core.models import Conversation, Message
from core.security import get_current_user_id
from schema.chat import ChatRequest, ChatResponse, ChatPendingResponse

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


def _safe_interrupt_payload(result: dict) -> dict | None:
    interrupt_block = result.get("__interrupt__") or []
    if not interrupt_block:
        return None

    payload = interrupt_block[0].value if hasattr(
        interrupt_block[0], "value") else interrupt_block[0]
    if isinstance(payload, dict):
        nested = payload.get("value") if "value" in payload and isinstance(
            payload.get("value"), dict) else payload
        return nested
    return {"summary": str(payload)}


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
    snapshot_values = (snapshot.values or {}) if snapshot is not None else {}
    tasks = getattr(snapshot, "tasks", []) or []
    is_paused = bool(getattr(snapshot, "next", None)) and any(
        getattr(task, "interrupts", None) for task in tasks
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
            "dataset_id": snapshot_values.get("dataset_id"),
        }
        result = await graph.ainvoke(follow_up_state, config=config)

    interrupted = "__interrupt__" in result
    interrupt_payload = _safe_interrupt_payload(
        result) if interrupted else None

    # What actually gets logged as the "assistant" turn — prefer the current
    # interrupt, then a current direct answer, then a generic note. Never use
    # a stale direct answer alongside a new interrupt.
    if interrupted:
        assistant_message_content = interrupt_payload.get(
            "summary", "Waiting for your input.") if interrupt_payload else "Waiting for your input."
    elif result.get("direct_answer"):
        assistant_message_content = result["direct_answer"]
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
        direct_answer=None if interrupted else result.get("direct_answer"),
    )


@router.get("/chat/{thread_id}/pending", response_model=ChatPendingResponse)
async def get_pending_decision(
    thread_id: str,
    request: Request,
    user_id: int = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
):
    await _get_owned_conversation(session, thread_id, user_id)

    graph = request.app.state.graph
    config = {"configurable": {"thread_id": thread_id}}

    snapshot = await graph.aget_state(config)
    is_paused = bool(snapshot.next) and any(
        task.interrupts for task in snapshot.tasks
    )

    interrupt_payload = None
    if is_paused and snapshot.tasks:
        for task in snapshot.tasks:
            if task.interrupts:
                interrupt_payload = task.interrupts[0].value
                break

    return ChatPendingResponse(
        interrupted=is_paused,
        interrupt=interrupt_payload,
    )
