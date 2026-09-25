# api/runs.py
import uuid

from fastapi import APIRouter, Depends, Request

from core.db import async_session
from core.models import Conversation, Message
from core.security import get_current_user_id
from schema.runs import RunRequest, RunResponse
from services.dataset_service import dataset_service
from services.llm_config import resolve_user_llm_config
from services.runtime_state import runtime_state_store
from cache.dataset_cache import dataset_summary_cache
from analysis.inspector import dataset_inspector
from agents.planner import planner_agent

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

    # --- Planning now happens HERE, before any graph is chosen ---
    # This is the one structural change needed to support multiple graph
    # families: we need to know problem_type BEFORE calling .ainvoke() on
    # anything, since that determines which compiled graph to use. So the
    # dataset load + inspection + planning that used to happen inside
    # dataset_node/planner_node now happens directly in this endpoint, and
    # the resulting analysis_plan is handed to the graph pre-built.
    dataframe = dataset_service.load_dataset(payload.dataset_id)

    dataset_summary = dataset_summary_cache.get(payload.dataset_id)
    if dataset_summary is None:
        dataset_summary = dataset_inspector.inspect(dataframe)
        dataset_summary_cache.set(payload.dataset_id, dataset_summary)

    async with async_session() as session:
        llm_config = await resolve_user_llm_config(session, user_id)

    plan = await planner_agent.plan(payload.user_query, dataset_summary, llm_config)

    # THE dispatch decision. Every future message on this thread must be
    # routed to the SAME graph — this is why pipeline_family is persisted
    # on the Conversation row below, not just used once here.
    family = "unsupervised" if plan.problem_type == "clustering" else "supervised"
    graph = (
        request.app.state.unsupervised_graph
        if family == "unsupervised"
        else request.app.state.graph
    )

    # Make the loaded dataframe available to the graph's runtime cache the
    # same way dataset_node normally would, since dataset_node is being
    # skipped for this pre-planned path (see route_intent's
    # "extract_constraints" branch in graphs/nodes.py).
    runtime_state_store.set_dataset(payload.dataset_id, dataframe)

    # A NEW thread_id per run — this is what makes it "fresh new id"
    # rather than reusing dataset_id, so two runs on the same dataset
    # (or two different users) never collide.
    thread_id = str(uuid.uuid4())

    initial_state = {
        "user_query": payload.user_query,
        "dataset_id": payload.dataset_id,
        "user_id": user_id,
        "llm_config": llm_config,
        "dataframe": None,
        "dataset_summary": dataset_summary,
        "analysis_plan": plan,
        "target_column": plan.target_column,
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
            pipeline_family=family,
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
