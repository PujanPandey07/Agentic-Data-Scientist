# api/chat.py — full updated file
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from langgraph.types import Command
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.db import get_db_session
from core.models import Conversation, Message
from core.security import get_current_user_id
from schema.chat import ChatRequest, ChatResponse, ChatPendingResponse

router = APIRouter(prefix="/api", tags=["Chat"])
logger = logging.getLogger(__name__)

STAGE_LABELS = {
    "cleaning": "Data cleaning",
    "eda": "Exploratory data analysis",
    "visualization": "Visualization",
    "feature_engineering": "Feature engineering",
    "model_selection": "Model training",
    "hyperparameter_tuning": "Hyperparameter tuning",
    "evaluation": "Evaluation",
    "reporting": "Reporting",
}


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


def _graph_for(request: Request, conversation: Conversation):
    """Every future message on a thread must go to the SAME compiled
    graph it started on — pipeline_family is persisted on the Conversation
    row at creation time (api/runs.py) specifically so this lookup is
    possible on every subsequent message, including resumes."""
    if getattr(conversation, "pipeline_family", "supervised") == "unsupervised":
        return request.app.state.unsupervised_graph
    return request.app.state.graph


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


def _stage_highlight(stage: str, state: dict) -> str | None:
    """One-line result summary for a completed stage, pulled from the same
    report fields reporting_node already relies on. Returns None for
    stages we don't have a specific highlight for — they still get
    listed as completed, just without extra detail."""
    if stage == "cleaning":
        report = state.get("cleaning_report") or {}
        if report:
            return (
                f"{report.get('original_rows')} rows -> {report.get('final_rows')} rows "
                f"({report.get('duplicates_removed')} duplicates removed)"
            )
    elif stage == "model_selection":
        report = state.get("training_report") or {}
        if report:
            return f"best model: {report.get('best_algorithm')}, CV score: {report.get('best_mean_cv_score')}"
    elif stage == "hyperparameter_tuning":
        report = state.get("hyperparameter_tuning_report") or {}
        if report and report.get("best_trial_score") is not None:
            return f"tuned score: {report.get('best_trial_score')}"
    elif stage == "evaluation":
        report = state.get("evaluation_report") or {}
        metrics = report.get("metrics") if report else None
        if metrics:
            return ", ".join(f"{k}: {v}" for k, v in metrics.items())
    return None


def _summarize_completed_stages(state: dict) -> str:
    """Human-readable list of what finished successfully before a crash,
    built from completed_tasks — the same field reporting_node uses to
    know what actually ran. No separate tracking needed: LangGraph only
    commits a stage's results to the checkpoint once that stage's node
    returns, so anything crash-adjacent that never completed simply
    won't be in completed_tasks or in its report field yet."""
    completed = state.get("completed_tasks") or []
    if not completed:
        return "No stages completed successfully before the failure."

    lines = []
    for stage in completed:
        label = STAGE_LABELS.get(stage, stage)
        highlight = _stage_highlight(stage, state)
        if highlight:
            lines.append(f"✅ {label} — {highlight}")
        else:
            lines.append(f"✅ {label} — completed")
    return "\n".join(lines)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    user_id: int = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
):
    conversation = await _get_owned_conversation(session, payload.thread_id, user_id)

    graph = _graph_for(request, conversation)
    config = {"configurable": {"thread_id": payload.thread_id}}

    snapshot = await graph.aget_state(config)
    snapshot_values = (snapshot.values or {}) if snapshot is not None else {}
    tasks = getattr(snapshot, "tasks", []) or []
    is_paused = bool(getattr(snapshot, "next", None)) and any(
        getattr(task, "interrupts", None) for task in tasks
    )
    active_interrupt_type = None
    for task in tasks:
        if getattr(task, "interrupts", None):
            val = getattr(task.interrupts[0], "value", None)
            if isinstance(val, dict):
                active_interrupt_type = val.get("type")
            break
    is_job_resume = (active_interrupt_type == "job_status")

    # Captured BEFORE the call — this is which node was about to run this
    # step. If the call crashes, this tells us where it crashed, since we
    # can't get that from an exception that gives no node context.
    attempted_stage = snapshot.next[0] if getattr(
        snapshot, "next", None) else None

    try:
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
            # Only log user turn if this was a human decision, not an automated background job resume
            if is_job_resume:
                user_message_content = None
            else:
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
                "user_id": user_id,
            }
            result = await graph.ainvoke(follow_up_state, config=config)

    except HTTPException:
        # Our own deliberate 400s above — not a graph crash, let them
        # propagate as-is.
        raise

    except Exception:
        # Any unexpected crash inside the graph (a node raising instead of
        # returning/interrupting cleanly). Without this, the checkpoint is
        # left pointing at whatever interrupt/task was pending when it
        # crashed, and every future message on this thread gets wrongly
        # routed as a "decision" forever — the conversation is permanently
        # stuck. Recovery: force the checkpoint into a clean, non-paused
        # state so the NEXT message is treated as a normal fresh
        # user_query.
        logger.exception(
            f"Graph execution crashed for thread {payload.thread_id} "
            f"(attempted stage: {attempted_stage}) — forcing recovery"
        )

        # Re-fetch state AFTER the crash. LangGraph commits each stage's
        # results to the checkpoint as soon as that stage's node returns —
        # a crash in a LATER stage doesn't undo earlier commits. So
        # anything that finished before the crash (cleaning/eda/training
        # reports, completed_tasks) is still here even though this call
        # itself failed.
        post_crash_snapshot = await graph.aget_state(config)
        post_crash_values = (
            post_crash_snapshot.values or {}) if post_crash_snapshot else {}

        completed_summary = _summarize_completed_stages(post_crash_values)
        stage_label = STAGE_LABELS.get(
            attempted_stage, attempted_stage or "a step")

        error_message = (
            f"Here's what completed before the pipeline ran into a problem:\n\n"
            f"{completed_summary}\n\n"
            f"❌ {stage_label} failed unexpectedly and the run stopped there. "
            f"You can ask about the results above, or try your request again."
        )

        recovery_values = {
            "direct_answer": error_message,
            # Clear job-tracking fields so a crash mid-background-job
            # can't leave a stale job_id hanging around either.
            "_training_job_id": None,
            "_tuning_job_id": None,
        }

        # as_node="direct_answer" tells LangGraph "treat this update as if
        # direct_answer_node had just run and returned these values" —
        # this clears whatever task/interrupt was pending. Relies on
        # direct_answer_node routing to END in the graph builder, so the
        # thread ends up NOT paused afterward. Report fields already in
        # post_crash_values (cleaning_report etc.) are untouched by this
        # update — only the keys listed above are overwritten.
        await graph.aupdate_state(config, recovery_values, as_node="direct_answer")

        session.add(Message(
            conversation_id=conversation.id,
            role="user",
            content=payload.user_query or "(decision)",
        ))
        session.add(Message(
            conversation_id=conversation.id,
            role="assistant",
            content=error_message,
        ))
        await session.commit()

        return ChatResponse(
            interrupted=False,
            interrupt=None,
            intent=None,
            direct_answer=error_message,
        )

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

    if user_message_content is not None:
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
    conversation = await _get_owned_conversation(session, thread_id, user_id)

    graph = _graph_for(request, conversation)
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
