# api/runs.py
import uuid

from fastapi import APIRouter, Depends, Request

from core.db import get_db_session
from core.models import Conversation, Message
from core.security import get_current_user_id
from schema.runs import RunRequest, RunResponse
from services.dataset_service import dataset_service

router = APIRouter(prefix="/api", tags=["Runs"])


def _conversation_title(user_query: str) -> str:
    """Create a readable history label without another LLM call."""
    title = " ".join(user_query.split())
    return title[:77].rstrip() + "..." if len(title) > 80 else title


@router.post("/runs", response_model=RunResponse)
async def create_run(
    payload: RunRequest,
    request: Request,
    user_id: int = Depends(get_current_user_id),
):
    # Fail fast with a clean 404 if the dataset_id doesn't map to a
    # real uploaded file, instead of letting the graph fail deep
    # inside some later node with a confusing error.
    dataset_service.get_dataset_path(payload.dataset_id)

    # A NEW thread_id per run — this is what makes it "fresh new id"
    # rather than reusing dataset_id, so two runs on the same dataset
    # (or two different users) never collide.
    thread_id = str(uuid.uuid4())

    # Mirrors test_graph.py's is_first_run=True branch exactly — every
    # GraphState field needs an initial value since this OVERWRITES
    # the checkpoint for a brand new thread_id.
    initial_state = {
        "user_query": payload.user_query,
        "dataset_id": payload.dataset_id,
        "dataframe": None,
        "dataset_summary": None,
        "analysis_plan": None,
        "cleaning_report": None,
        "eda_report": None,
        "visualization_plan": None,
        "visualization_results": None,
        "feature_engineering_plan": None,
        "feature_engineering_report": None,
        "current_task": None,
        "remaining_tasks": [],
        "completed_tasks": [],
        "execution_logs": [],
        "intent": None,
        "direct_answer": None,
    }

    # request.app.state.graph — the SAME compiled graph object created
    # once in lifespan, reused across every request. No recompiling here.
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": thread_id}}

    result = await graph.ainvoke(initial_state, config=config)

    # Record this run in OUR OWN db (separate from LangGraph's
    # checkpoint store, which has no concept of "list all runs" or
    # "which user owns this").
    async with request.app.state.db_session() as session:
        conversation = Conversation(
            thread_id=thread_id,
            dataset_id=payload.dataset_id,
            user_id=user_id,
            title=_conversation_title(payload.user_query),
        )
        session.add(conversation)
        await session.flush()  # populates conversation.id before we reference it below

        session.add(Message(
            conversation_id=conversation.id,
            role="user",
            content=payload.user_query,
        ))
        await session.commit()

    interrupted = "__interrupt__" in result
    interrupt_payload = result["__interrupt__"][0].value if interrupted else None

    return RunResponse(
        thread_id=thread_id,
        interrupted=interrupted,
        interrupt=interrupt_payload,
        intent=result.get("intent"),
        direct_answer=result.get("direct_answer"),
    )
