# api/chat.py
from fastapi import APIRouter, HTTPException, Request
from langgraph.types import Command

from schema.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/api", tags=["Chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request):
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": payload.thread_id}}

    # Ask the graph itself: is this thread currently paused on an
    # interrupt? aget_state() returns a snapshot of the checkpoint —
    # .next tells us which node(s) would run next. If the graph is
    # paused, .next points at the SAME node that raised interrupt(),
    # waiting for a resume — non-empty .next + a pending interrupt
    # is exactly what "paused" means here.
    snapshot = await graph.aget_state(config)
    is_paused = bool(snapshot.next) and any(
        task.interrupts for task in snapshot.tasks
    )

    if is_paused:
        # We're paused — this message MUST be a decision, not a query.
        if payload.decision is None:
            raise HTTPException(
                status_code=400,
                detail="This conversation is waiting for a decision (approve/reject/edit), not a new message.",
            )
        resume_payload = {
            "approved": payload.decision.approved,
            "edit_instruction": payload.decision.edit_instruction,
        }
        result = await graph.ainvoke(Command(resume=resume_payload), config=config)
    else:
        # Not paused — this must be a normal follow-up message.
        if payload.user_query is None:
            raise HTTPException(
                status_code=400,
                detail="user_query is required when the conversation isn't waiting on a decision.",
            )
        # Mirrors test_graph.py's is_first_run=False branch — only
        # user_query + dataset_id, everything else stays as the
        # checkpoint already has it.
        follow_up_state = {
            "user_query": payload.user_query,
            "dataset_id": snapshot.values.get("dataset_id"),
        }
        result = await graph.ainvoke(follow_up_state, config=config)

    interrupted = "__interrupt__" in result
    interrupt_payload = result["__interrupt__"][0].value if interrupted else None

    return ChatResponse(
        interrupted=interrupted,
        interrupt=interrupt_payload,
        intent=result.get("intent"),
        direct_answer=result.get("direct_answer"),
    )
