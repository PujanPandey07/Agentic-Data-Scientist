from services.llm_config import resolve_user_llm_config
from core.db import async_session
from services.job_queue import enqueue_training_job, check_job_result
import logging
from sklearn.model_selection import train_test_split as sk_train_test_split

from utilis.constraints import format_constraints, detect_forced_algorithm
from schema.model_selection import ModelCandidate, ModelSelectionPlan
from langgraph.types import interrupt
from agents.constraint_extractor import constraints_extractor_agent
from .plan_validation import validate_plan, extract_explicit_target_column

from agents.refine_target import refine_target_agent
from llm.provider import get_llm
from agents.intent_router import intent_router_agent
from services.hyper_parameters_tuning import HyperparameterTuningService
from services.reporting import ReportingService
from services.analysis_context import analysis_context_builder
from cache.analysis_cache import analysis_context_cache
from cache.dataset_cache import dataset_summary_cache
from services.short_term_memory import short_term_memory_manager
from services.long_term_memory import long_term_memory_manager
from services.context_assembler import llm_context_assembler
from services.evaluation import EvaluationService
from services.trainning import TrainingService
from services.runtime_state import runtime_state_store
from agents.model_selection_planner import ModelSelectionPlannerAgent
import os
from services.feature_engineering import FeatureEngineeringService
from agents.feature_engineering_planner import FeatureEngineeringPlannerAgent
from services.visualization_service import visualization_service
from langgraph.graph import END
from services.eda_service import eda_service
from services.cleaning_service import cleaning_service
from agents.planner import planner_agent
from services.dataset_service import dataset_service
from analysis.inspector import dataset_inspector
from services.executiopn_service import execution_service
from agents.visualization_planner import visualization_planner_agent
from services.job_queue import enqueue_training_job, check_job_result, enqueue_tuning_job
import pandas as pd

logger = logging.getLogger(__name__)

CASCADE_MAP = {
    "cleaning": ["eda", "visualization", "feature_engineering", "model_selection", "hyperparameter_tuning", "evaluation", "reporting"],
    "eda": ["visualization", "feature_engineering", "model_selection", "hyperparameter_tuning", "evaluation", "reporting"],
    "visualization": ["reporting"],
    "feature_engineering": ["model_selection", "hyperparameter_tuning", "evaluation", "reporting"],
    "model_selection": ["hyperparameter_tuning", "evaluation", "reporting"],
    "hyperparameter_tuning": ["evaluation", "reporting"],
    "evaluation": ["reporting"],
    "reporting": [],
}
PIPELINE_SYSTEM_CONTEXT = """
You are the advisory assistant for an automated data-science pipeline system.

IMPORTANT — the system knowledge below exists so you can accurately answer questions ABOUT this system when the user asks them (how it works, what it can do, what happens at a given stage, how to phrase a request, etc.). It is background knowledge, not a script to follow on every turn. For any message that isn't asking about this system or how to use it — general knowledge questions, questions about you as an assistant, casual chat, or a dataset question that doesn't concern the pipeline — answer that question directly and normally. Do NOT mention pipeline stages, do NOT produce an example prompt, and do NOT redirect the conversation toward running an analysis. Answering plainly and stopping there is correct and expected for most messages.

How this system actually works:
- The user's message becomes a query that a planner turns into a plan: a detected problem_type (classification or regression), a target_column, and an ordered list of pipeline stages to run. The user reviews and can approve or edit this plan before anything runs.
- The stages, always run in this order, are: cleaning -> eda -> visualization -> feature_engineering -> model_selection -> hyperparameter_tuning -> evaluation -> reporting.
- The user can embed per-stage constraints directly in their prompt, e.g. "for feature engineering, one-hot encode the categorical columns", "for model selection, use XGBoost", or "skip visualization" — these are extracted automatically and applied at the matching stage.
- Model selection and hyperparameter tuning run as background jobs; the user sees a "running in the background" status while these complete, rather than the chat blocking.
- After a full run completes, the user gets a final report with conclusions, plus charts and an exportable PDF.
- The user can ask to REFINE a specific completed stage afterward (e.g. "try a different algorithm", "add a chart of X") — this reruns just that stage and everything downstream of it, not the whole pipeline from scratch.
- If a run is interrupted partway (a crash, or the user leaving mid-run), the user can say "continue" or "retry" to resume from where it left off, rather than starting over.
- The user does NOT need to write code or specify implementation details — the system's own deterministic services execute each stage. What matters is that the target column and problem type are clear (or inferable), and that any real preferences are stated explicitly.
- Users can optionally add their own API key (OpenAI, Anthropic, or Gemini) in settings so their runs use their own model instead of the shared default — this only affects which LLM plans/reasons about the data, not the underlying pipeline mechanics.

When (and only when) the user is asking for the best/ideal prompt for their dataset:
1. Inspect the real dataset summary given to you (column names, types, stats) to identify the most likely target column and whether the problem is classification or regression. If it's genuinely ambiguous, say so briefly and ask the user to confirm the target column rather than guessing silently.
2. Produce an actual example prompt, clearly set off (e.g. in a quoted block), that the user could copy and paste directly into this chat to start a run. It should explicitly name:
   - The target column
   - The problem type (classification or regression)
   - Any stages worth emphasizing, skipping, or constraining, only if there's a real reason based on THIS dataset (e.g. "skip visualization" for a very wide dataset, or a stratification note for imbalanced classes) — don't pad it with generic advice that applies to every dataset.
3. Keep the example prompt itself short and natural — one or two sentences a real user would actually type, not a formal spec. Never invent pipeline stages, config options, or capabilities beyond what's listed above.
"""


def _stringify_llm_content(content) -> str:
    """LangChain's .content is normally a plain string, but some providers
    (seen with gemini-3.5-flash-lite) return a list of content blocks
    instead (e.g. [{'type': 'text', 'text': '...'}, {'type': 'thought_signature', ...}]).
    Flatten to plain text so downstream code that expects a string (chat
    memory, direct_answer state) never chokes on a list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if text:
                    parts.append(text)
        return "".join(parts)
    return str(content)


def advance_execution(state, task_name: str, message: str, status: str = "success"):
    """
    status="success" -> task actually ran and is recorded in completed_tasks.
    status="skipped" -> task was skipped (e.g. missing inputs); logged but
    NOT added to completed_tasks, so history reflects what really executed.
    """
    print(f"Running {task_name} node...")

    execution_service.advance_task(
        state=state,
        completed_task=task_name,
        message=message,
        status=status,
    )

    return state


# State keys produced by each stage. Cleared when the stage (or anything
# upstream that regenerates it) is about to rerun, so stale results never
# leak across pipeline runs or refinements.
STAGE_FIELDS = {
    "cleaning": ["cleaning_report"],
    "eda": ["eda_report"],
    "visualization": ["visualization_plan", "visualization_results", "_viz_just_completed"],
    "feature_engineering": ["feature_engineering_plan", "feature_engineering_report", "train_df", "test_df", "_fe_just_completed"],
    "model_selection": ["model_selection_plan", "training_report", "trained_model_path"],
    "hyperparameter_tuning": ["hyperparameter_tuning_report", "tuned_model_path"],
    "evaluation": ["evaluation_report"],
    "reporting": ["final_report"],
}


def _clear_stale_fields(state, from_stage: str):
    """Remove all results produced by `from_stage` and every downstream stage.
    Called before re-executing part of the pipeline so old artifacts from a
    previous run (or a prior refine) don't leak forward.
    """
    stages_to_clear = [from_stage] + CASCADE_MAP.get(from_stage, [])
    cleared = []
    for stage in stages_to_clear:
        for key in STAGE_FIELDS.get(stage, []):
            if key in state and state.get(key) is not None:
                state[key] = None
                cleared.append(key)
    if cleared:
        logger.info(
            f"Cleared stale fields for stages {stages_to_clear}: {cleared}")


def _runtime_dataframe(state):
    dataframe = state.get("dataframe")
    if dataframe is not None:
        return dataframe
    return runtime_state_store.get_dataset(state.get("dataset_id"))


def _set_runtime_dataframe(state, dataframe):
    dataset_id = state.get("dataset_id")
    if dataset_id:
        runtime_state_store.set_dataset(dataset_id, dataframe)
    state["dataframe"] = None


def _runtime_train_test(state):
    train_df = state.get("train_df")
    test_df = state.get("test_df")
    if train_df is not None or test_df is not None:
        return train_df, test_df
    dataset_id = state.get("dataset_id")
    return runtime_state_store.get_train_test(dataset_id)


def _set_runtime_train_test(state, train_df, test_df):
    dataset_id = state.get("dataset_id")
    if dataset_id:
        runtime_state_store.set_train_test(dataset_id, train_df, test_df)
    state["train_df"] = None
    state["test_df"] = None


def _ensure_train_test_split(state, test_size: float = 0.2, random_state: int = 42):
    """Stratified (classification) or plain (regression) 80/20 train/test
    split, stored in the runtime cache rather than the checkpointed graph state.
    """
    train_df, test_df = _runtime_train_test(state)
    if train_df is not None and test_df is not None:
        return

    df = _runtime_dataframe(state)
    target_column = state.get("target_column")
    analysis_plan = state.get("analysis_plan")
    problem_type = analysis_plan.problem_type if analysis_plan else "classification"

    if df is None or target_column is None or target_column not in df.columns:
        return  # let the caller's own missing-input guard handle it

    stratify_col = None
    if problem_type == "classification":
        counts = df[target_column].value_counts()
        if (counts >= 2).all():
            stratify_col = df[target_column]

    try:
        train_df, test_df = sk_train_test_split(
            df, test_size=test_size, random_state=random_state,
            stratify=stratify_col,
        )
    except ValueError as e:
        logger.warning(
            f"Stratified split failed ({e}), falling back to unstratified split")
        train_df, test_df = sk_train_test_split(
            df, test_size=test_size, random_state=random_state,
        )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    _set_runtime_train_test(state, train_df, test_df)

    logger.info(
        f"Train/test split created: train={train_df.shape}, test={test_df.shape}"
    )


async def dataset_node(state):
    dataset_id = state["dataset_id"]
    dataframe = dataset_service.load_dataset(dataset_id)
    summary = dataset_summary_cache.get(dataset_id)
    if summary is None:
        summary = dataset_inspector.inspect(dataframe)
        dataset_summary_cache.set(dataset_id, summary)

    _set_runtime_dataframe(state, dataframe)
    state["dataset_summary"] = summary
    state.pop("dataframe", None)

    return state


async def planner_node(state):

    plan = await planner_agent.plan(
        state["user_query"],
        state["dataset_summary"],
        state.get("llm_config"),
    )

    print("\n========== PLANNER RESULT ==========")
    print(plan)
    print("====================================\n")

    state["analysis_plan"] = plan
    state["target_column"] = plan.target_column

    return state


async def initialize_execution_node(state):
    plan = state.get("analysis_plan")
    tasks = plan.tasks if plan else []

    state["current_task"] = tasks[0] if tasks else None
    state["remaining_tasks"] = tasks[1:] if len(tasks) > 1 else []
    state["completed_tasks"] = []
    state["execution_logs"] = [
        {
            "node": "initialize_execution",
            "message": "Execution queue initialized."
        }
    ]

    # Fresh pipeline — clear any artifacts left from a previous run on this
    # thread so stale results never leak forward.
    _clear_stale_fields(state, "cleaning")

    return state


async def router(state):
    viz_done = bool(state.get("_viz_just_completed"))
    fe_done = bool(state.get("_fe_just_completed"))
    completing_fan_out = bool(state.get("fan_out_viz_fe"))

    new_completed = []
    new_logs = []

    if viz_done:
        new_completed.append("visualization")
        new_logs.append({"node": "visualization", "status": "success",
                        "message": state["_viz_just_completed"]})
        state["_viz_just_completed"] = None

    if fe_done:
        new_completed.append("feature_engineering")
        new_logs.append({"node": "feature_engineering",
                        "status": "success", "message": state["_fe_just_completed"]})
        state["_fe_just_completed"] = None

    if new_completed:
        state["completed_tasks"] = state.get(
            "completed_tasks", []) + new_completed
        state["execution_logs"] = state.get("execution_logs", []) + new_logs

    current_task = state.get("current_task")
    remaining = state.get("remaining_tasks", [])

    both_pending = (
        (current_task == "visualization" and "feature_engineering" in remaining) or
        (current_task == "feature_engineering" and "visualization" in remaining)
    )

    if both_pending:
        # Dispatching the fan-out NOW — pre-advance current_task past
        # both, since route_task uses fan_out_viz_fe (not current_task)
        # to route this case.
        new_remaining = [t for t in remaining if t not in (
            "visualization", "feature_engineering")]
        state["current_task"] = new_remaining[0] if new_remaining else None
        state["remaining_tasks"] = new_remaining[1:] if len(
            new_remaining) > 1 else []
        state["fan_out_viz_fe"] = True
    elif (viz_done or fe_done) and not completing_fan_out:
        # Standalone completion (viz/fe ran alone, not as a fan-out
        # pair) — current_task was NOT pre-advanced for this case, so
        # advance past it now.
        state["current_task"] = remaining[0] if remaining else None
        state["remaining_tasks"] = remaining[1:] if len(remaining) > 1 else []
        state["fan_out_viz_fe"] = False
    else:
        # Either nothing completed this call, OR we just finished
        # completing a fan-out pair whose current_task was ALREADY
        # advanced at dispatch time — don't advance again, just reset
        # the flag.
        state["fan_out_viz_fe"] = False

    return state


def route_task(state):
    if state.get("fan_out_viz_fe"):
        return ["visualization_planner", "feature_engineering_planner"]

    task = state.get("current_task")

    if task == "cleaning":
        return "cleaning"
    if task == "eda":
        return "eda"
    if task == "feature_engineering":
        return "feature_engineering"
    if task == "model_selection":
        return "model_selection"
    if task == "visualization":
        return "visualization"
    if task == "hyperparameter_tuning":
        return "hyperparameter_tuning"
    if task == "evaluation":
        return "evaluation"
    if task == "reporting":
        return "reporting"

    return END


async def cleaning_node(state):

    dataframe = _runtime_dataframe(state)
    target_column = state.get("target_column")

    if dataframe is None:
        raise ValueError("Dataframe not found in graph state.")

    cleaned_dataframe, report = cleaning_service.clean_dataset(
        dataframe
    )

    _set_runtime_dataframe(state, cleaned_dataframe)
    if target_column is not None:
        clean_target = cleaning_service.standardize_column_name(target_column)
        state["target_column"] = clean_target
        if state.get("analysis_plan") is not None:
            state["analysis_plan"].target_column = clean_target

    summary = state.get("dataset_summary")
    if summary is not None:
        if hasattr(summary, "column_names") and summary.column_names:
            summary.column_names = [cleaning_service.standardize_column_name(
                c) for c in summary.column_names]
        if hasattr(summary, "numerical_columns") and summary.numerical_columns:
            summary.numerical_columns = [cleaning_service.standardize_column_name(
                c) for c in summary.numerical_columns]
        if hasattr(summary, "categorical_columns") and summary.categorical_columns:
            summary.categorical_columns = [cleaning_service.standardize_column_name(
                c) for c in summary.categorical_columns]

    state["cleaning_report"] = report
    # Data changed — any cached train/test split is now stale.
    state["train_df"] = None
    state["test_df"] = None
    runtime_state_store.clear_train_test(state.get("dataset_id"))

    print("REPORT IN STATE:", state.get("cleaning_report"))

    return advance_execution(
        state,
        "cleaning",
        "Dataset cleaning completed successfully."
    )


async def eda_node(state):
    dataframe = _runtime_dataframe(state)
    if dataframe is None:
        raise ValueError("Dataframe not found in graph state.")
    eda_report = eda_service.analyze(dataframe)

    state["eda_report"] = eda_report

    return advance_execution(
        state,
        "eda",
        "Exploratory data analysis completed successfully."
    )


# ---------------------------------------------------------------------------
# NOTE: visualization_planner_node and feature_engineering_planner_node are
# each defined ONCE here — these are the versions that must run for the
# fan-out to work. They return partial-update dicts (only the key they
# actually set), NOT the whole state, because when both run concurrently in
# the same step, returning the full state causes both branches to "write"
# to every key (including untouched ones like user_query), which LangGraph
# rejects with InvalidUpdateError. A second, older copy of each of these
# functions previously existed further down in this file — Python silently
# used whichever definition came last, which is why the fix appeared not to
# take effect. Do not add a second definition of either function again.
# ---------------------------------------------------------------------------

async def visualization_planner_node(state):

    user_query = state.get("user_query")
    dataset_summary = state.get("dataset_summary")
    eda_report = state.get("eda_report")

    if not user_query:
        raise ValueError("User query not found in graph state.")

    if dataset_summary is None:
        raise ValueError("Dataset summary not found in graph state.")

    if eda_report is None:
        raise ValueError("EDA report not found in graph state.")

    visualization_plan = await visualization_planner_agent.plan_visualizations(
        user_query=user_query,
        dataset_summary=dataset_summary,
        eda_report=eda_report,
        constraints=(state.get("user_constraints")
                     or {}).get("visualization", []),
        llm_config=state.get("llm_config"),
    )

    # Partial-update return — NOT the whole state. Two nodes running
    # concurrently must each touch only their own key, or LangGraph
    # can't merge them (InvalidUpdateError).
    return {"visualization_plan": visualization_plan}


async def model_selection_planner_node(state):
    """LLM reasoning: decides WHICH models to try.

    If a forced algorithm is detected in user constraints, skip the LLM
    call entirely and build a deterministic plan with sensible defaults.
    """
    constraints = (state.get("user_constraints")
                   or {}).get("model_selection", [])

    # ------------------------------------------------------------------
    # FAST PATH: user explicitly named an algorithm — skip the LLM entirely
    # ------------------------------------------------------------------
    forced = detect_forced_algorithm(constraints)
    if forced:
        analysis_plan = state.get("analysis_plan")
        problem_type = analysis_plan.problem_type if analysis_plan else "classification"

        scoring_metric = "accuracy" if problem_type == "classification" else "neg_mean_squared_error"

        plan = ModelSelectionPlan(
            strategy="standard",
            sample_size=None,
            candidates=[
                ModelCandidate(
                    algorithm=forced,
                    reason=f"User explicitly requested '{forced}' via constraint.",
                    estimated_time_seconds=30,
                    hyperparams={},
                    priority=1,
                )
            ],
            cv_folds=5,
            scoring_metric=scoring_metric,
            time_budget_minutes=5,
            notes=[
                f"Algorithm forced by user constraint: '{forced}'. "
                "LLM planner skipped to save tokens and avoid rate limits."
            ],
            forced_algorithm=forced,
        )

        logger.info(
            f"Forced algorithm '{forced}' detected — skipping model-selection LLM call"
        )
        state["model_selection_plan"] = plan
        return state

    # ------------------------------------------------------------------
    # NORMAL PATH: no forced algorithm — ask the LLM to choose candidates
    # ------------------------------------------------------------------
    agent = ModelSelectionPlannerAgent()
    plan = await agent.plan(
        user_query=state.get("user_query", ""),
        dataset_summary=state.get("dataset_summary"),
        eda_report=state.get("eda_report", {}),
        feature_engineering_report=state.get("feature_engineering_report"),
        constraints=constraints,
        llm_config=state.get("llm_config"),
    )

    # Safety net: if the LLM somehow missed the constraint, enforce it here
    forced = detect_forced_algorithm(constraints)
    if forced:
        existing = next(
            (c for c in plan.candidates if c.algorithm == forced), None)
        plan.forced_algorithm = forced
        plan.candidates = [
            existing if existing else ModelCandidate(
                algorithm=forced,
                reason="User explicitly requested this algorithm.",
                estimated_time_seconds=30,
                hyperparams={},
                priority=1,
            )
        ]
        logger.info(
            f"Forced algorithm '{forced}' detected post-LLM — overriding candidates"
        )

    state["model_selection_plan"] = plan
    return state


async def visualization_node(state):
    dataframe = _runtime_dataframe(state)
    if dataframe is None:
        raise ValueError("Dataframe not found in graph state.")

    visualization_plan = state.get("visualization_plan")
    if visualization_plan is None:
        raise ValueError("Visualization plan not found in graph state.")

    visualizations = visualization_service.generate_visualizations(
        dataframe=dataframe,
        visualization_plan=visualization_plan,
        dataset_id=state.get("dataset_id"),
    )

    print("Running visualization node...")

    return {
        "visualization_results": visualizations,
        "_viz_just_completed": "Visualizations generated successfully.",
    }


async def feature_engineering_planner_node(state):
    """LLM reasoning: decides WHAT feature engineering to do."""
    agent = FeatureEngineeringPlannerAgent()

    plan = await agent.plan(
        user_query=state.get("user_query", ""),
        dataset_summary=state.get("dataset_summary"),
        eda_report=state.get("eda_report", {}),
        target_column=state.get("target_column"),
        constraints=(state.get("user_constraints") or {}
                     ).get("feature_engineering", []),
        llm_config=state.get("llm_config"),
    )

    return {"feature_engineering_plan": plan}


async def feature_engineering_node(state):
    df = _runtime_dataframe(state)
    plan = state.get("feature_engineering_plan")
    target_column = state.get("target_column")

    print("Running feature_engineering node...")

    if df is None or plan is None:
        return {
            "feature_engineering_report": {
                "error": "Missing dataframe or feature engineering plan"
            },
            "_fe_just_completed": "Skipped: missing dataframe or feature engineering plan",
        }

    # IMPORTANT:
    # Split BEFORE feature engineering so that transformations
    # such as scaling, encoding, variance selection, correlation
    # selection, binning, and polynomial features learn only from train.
    split_state = dict(state)

    _ensure_train_test_split(split_state)

    train_df, test_df = _runtime_train_test(split_state)

    if train_df is None or test_df is None:
        return {
            "feature_engineering_report": {
                "error": "Failed to create train/test split before feature engineering"
            },
            "_fe_just_completed": "Skipped: failed to create train/test split before feature engineering",
        }

    service = FeatureEngineeringService()

    train_transformed, test_transformed, report = (
        service.apply_plan_train_test(
            train_df,
            test_df,
            plan,
            target_column=target_column,
        )
    )

    # Keep a combined dataframe for downstream state/report compatibility.
    transformed_df = pd.concat(
        [train_transformed, test_transformed],
        ignore_index=True,
    )

    msg = (
        f"Feature engineering complete. "
        f"Train: {train_df.shape} -> {train_transformed.shape}, "
        f"Test: {test_df.shape} -> {test_transformed.shape}. "
        f"Steps: {report['steps_executed']}/{len(plan.steps)} succeeded."
    )

    _set_runtime_dataframe(state, transformed_df)
    _set_runtime_train_test(state, train_transformed, test_transformed)

    return {
        "feature_engineering_report": report,
        "_fe_just_completed": msg,
    }


# ---------------------------------------------------------------------------
# TRAINING — split into enqueue + poll.
#
# WHY: interrupt() re-executes its ENCLOSING NODE FROM THE TOP on every
# resume (this is documented LangGraph behavior — "the graph resumes from
# the start of the node, re-executing all logic"). The original single
# training_node set state["_training_job_id"] = job_id AFTER enqueueing but
# BEFORE calling interrupt() — that assignment lives only in the node's
# local `state` dict and is never committed to the checkpoint, because the
# node never reached a real `return` before pausing. So on every resume,
# job_id read back out as None again, and the node enqueued a BRAND NEW job
# every single time — this was confirmed live: dozens of duplicate jobs
# fired in rapid succession for hyperparameter_tuning (same bug, more
# visible because those jobs finished fast).
#
# FIX: enqueue_training_node does the enqueue and returns immediately (a
# real graph step, which DOES commit to the checkpoint). poll_training_node
# is a separate node that only reads the now-reliably-persisted job_id and
# checks status — it's safe to let LangGraph re-run this one from the top
# on every resume, since it has no enqueue side effect of its own.
# ---------------------------------------------------------------------------

async def enqueue_training_node(state):
    dataset_id = state.get("dataset_id")
    plan = state.get("model_selection_plan")
    target_column = state.get("target_column")

    if plan is None or target_column is None:
        state["training_report"] = {
            "error": "Missing model selection plan or target column"}
        state["_training_job_id"] = None
        return advance_execution(
            state, "model_selection", "Skipped: missing required inputs",
            status="skipped",
        )

    job_id = await enqueue_training_job(
        dataset_id, target_column, plan.model_dump()
    )
    state["_training_job_id"] = job_id
    return state


async def poll_training_node(state):
    job_id = state.get("_training_job_id")

    if job_id is None:
        # enqueue_training_node already handled a skip case above and
        # advanced past model_selection itself — nothing to poll.
        return state

    outcome = await check_job_result(job_id)

    if outcome is None:
        interrupt({
            "type": "job_status",
            "job_type": "training",
            "job_id": job_id,
            "summary": "Training is running in the background — this may take a moment.",
        })
        return state

    state["_training_job_id"] = None

    if outcome["status"] == "failed":
        state["training_report"] = {"error": outcome["error"]}
        return advance_execution(
            state, "model_selection",
            f"Training failed: {outcome['error']}",
            status="skipped",
        )

    result = outcome["result"]
    state["training_report"] = result
    state["trained_model_path"] = result.get("model_path")

    msg = (
        f"Training complete. Best: {result['best_algorithm']} "
        f"with CV score {result['best_mean_cv_score']}. "
        f"Model saved to {result.get('model_path', 'N/A')}"
    )
    return advance_execution(state, "model_selection", msg)


async def evaluation_node(state):
    """Deterministic evaluation: loads the TUNED model (falling back to the
    untuned trained model if tuning was skipped/failed), scores it against
    the held-out TEST split only — never training data."""
    _, df = _runtime_train_test(state)
    target_column = state.get("target_column")
    model_path = state.get("tuned_model_path") or state.get(
        "trained_model_path")
    analysis_plan = state.get("analysis_plan")
    dataset_id = state.get("dataset_id")

    if df is None or target_column is None or model_path is None:
        state["evaluation_report"] = {
            "error": "Missing test dataframe, target column, or trained model path"
        }
        return advance_execution(
            state, "evaluation", "Skipped: missing required inputs",
            status="skipped",
        )

    if not os.path.exists(model_path):
        state["evaluation_report"] = {
            "error": f"Model file not found: {model_path}"
        }
        return advance_execution(
            state, "evaluation", "Skipped: model file missing",
            status="skipped",
        )

    problem_type = analysis_plan.problem_type if analysis_plan else "classification"

    service = EvaluationService()
    report = service.evaluate(
        df=df,
        target_column=target_column,
        model_path=model_path,
        problem_type=problem_type,
        dataset_id=dataset_id,
    )

    state["evaluation_report"] = report

    metrics = report["metrics"]
    if report["problem_type"] == "classification":
        msg = (
            f"Evaluation complete. Accuracy: {metrics['accuracy']}, "
            f"F1 (weighted): {metrics['f1_weighted']}"
        )
    else:
        msg = (
            f"Evaluation complete. RMSE: {metrics['rmse']}, "
            f"R²: {metrics['r2']}"
        )

    return advance_execution(state, "evaluation", msg)


async def reporting_node(state):
    """Aggregate all results into final report."""
    service = ReportingService()
    report = service.generate_report(state)

    state["final_report"] = report
    analysis_context = analysis_context_builder.build(state, report)
    analysis_context_cache.set(report["dataset_id"], analysis_context)
    conversation_id = state.get("conversation_id") or report.get("dataset_id")
    recommendation = (report.get("conclusions") or {}).get("recommendation")
    if conversation_id and recommendation and recommendation != "N/A":
        await long_term_memory_manager.upsert(
            conversation_id,
            "latest_analysis_conclusion",
            recommendation,
            importance=0.75,
        )

    msg = (
        f"Reporting complete. "
        f"Report saved to {report.get('report_path', 'N/A')}. "
        f"Best model: {report['conclusions']['best_model']} "
        f"(accuracy: {report['conclusions'].get('final_accuracy', 'N/A')})"
    )
    return advance_execution(state, "reporting", msg)


# ---------------------------------------------------------------------------
# HYPERPARAMETER TUNING — same enqueue/poll split as training, same reason.
# ---------------------------------------------------------------------------

async def enqueue_tuning_node(state):
    dataset_id = state.get("dataset_id")
    plan = state.get("model_selection_plan")
    training_report = state.get("training_report")
    target_column = state.get("target_column")
    analysis_plan = state.get("analysis_plan")

    if plan is None or training_report is None or target_column is None or dataset_id is None:
        state["hyperparameter_tuning_report"] = {
            "status": "skipped", "reason": "missing inputs"}
        state["tuned_model_path"] = state.get("trained_model_path")
        state["_tuning_job_id"] = None
        return advance_execution(
            state, "hyperparameter_tuning", "Skipped: missing inputs",
            status="skipped",
        )

    if plan.strategy == "quick":
        state["hyperparameter_tuning_report"] = {
            "status": "skipped", "reason": "quick strategy"}
        state["tuned_model_path"] = state.get("trained_model_path")
        state["_tuning_job_id"] = None
        return advance_execution(
            state, "hyperparameter_tuning", "Skipped: quick strategy",
            status="skipped",
        )

    best_algorithm = training_report.get("best_algorithm")
    best_candidate = next(
        (c for c in plan.candidates if c.algorithm == best_algorithm), None)
    if not best_candidate:
        state["hyperparameter_tuning_report"] = {
            "status": "skipped", "reason": "best candidate not found"}
        state["tuned_model_path"] = state.get("trained_model_path")
        state["_tuning_job_id"] = None
        return advance_execution(
            state, "hyperparameter_tuning", "Skipped: candidate not found",
            status="skipped",
        )

    problem_type = analysis_plan.problem_type if analysis_plan else "classification"

    job_id = await enqueue_tuning_job(
        dataset_id, target_column, best_candidate.model_dump(),
        problem_type,
        max_trials=20 if plan.strategy == "standard" else 50,
        time_budget_seconds=120 if plan.strategy == "standard" else 300,
    )
    state["_tuning_job_id"] = job_id
    return state


async def poll_tuning_node(state):
    job_id = state.get("_tuning_job_id")

    if job_id is None:
        # enqueue_tuning_node already handled a skip case above and
        # advanced past hyperparameter_tuning itself — nothing to poll.
        return state

    outcome = await check_job_result(job_id)

    if outcome is None:
        interrupt({
            "type": "job_status",
            "job_type": "hyperparameter_tuning",
            "job_id": job_id,
            "summary": "Hyperparameter tuning is running in the background — this may take a moment.",
        })
        return state

    state["_tuning_job_id"] = None

    if outcome["status"] == "failed":
        state["hyperparameter_tuning_report"] = {
            "status": "failed", "error": outcome["error"]}
        state["tuned_model_path"] = state.get("trained_model_path")
        return advance_execution(
            state, "hyperparameter_tuning",
            f"Tuning failed: {outcome['error']}",
            status="skipped",
        )

    result = outcome["result"]
    state["hyperparameter_tuning_report"] = result
    state["tuned_model_path"] = result["tuned_model_path"]

    top_param = max(result["param_importance"],
                    key=result["param_importance"].get) if result["param_importance"] else "N/A"
    msg = (
        f"Tuning complete. Score: {result['best_trial_score']}. "
        f"Trials: {result['num_trials_completed']}. "
        f"Top param: {top_param}"
    )
    return advance_execution(state, "hyperparameter_tuning", msg)


async def intent_router_node(state):
    # direct_answer is a response for one chat turn, not durable conversation
    # state. Clear it before classifying the next message so old answers cannot
    # leak into a new intent or resume response.
    state["direct_answer"] = None
    state["no_prior_analysis"] = False

    dataset_id = state.get("dataset_id")
    conversation_id = state.get("conversation_id") or dataset_id
    user_query = state.get("user_query", "")
    if conversation_id and user_query:
        await short_term_memory_manager.add_message(
            conversation_id, "user", user_query
        )
        await short_term_memory_manager.update_facts(
            conversation_id, current_goal=user_query
        )
        await long_term_memory_manager.extract_and_store(
            conversation_id, user_query
        )

    has_prior_report = (
        state.get("final_report") is not None
        or (
            dataset_id is not None
            and analysis_context_cache.get(dataset_id) is not None
        )
    )
    pipeline_in_progress = bool(
        state.get("current_task") or state.get("remaining_tasks")
    )

    classification = await intent_router_agent.classify(
        user_query=user_query,
        has_prior_report=has_prior_report,
        pipeline_in_progress=pipeline_in_progress,
        llm_config=state.get("llm_config"),
    )

    intent = classification.intent

    if intent in ("explain_result", "refine_step") and not has_prior_report:
        logger.warning(
            f"LLM returned '{intent}' with no prior report — overriding to general_question"
        )
        intent = "general_question"
        state["no_prior_analysis"] = True

    if intent == "resume_pipeline" and not pipeline_in_progress:
        logger.warning(
            f"LLM returned 'resume_pipeline' with no pipeline in progress — overriding to general_question"
        )
        intent = "general_question"
        state["no_prior_analysis"] = True

    state["intent"] = intent
    return state


async def direct_answer_node(state):
    llm = get_llm(state.get("llm_config"))
    conversation_id = state.get("conversation_id") or state.get("dataset_id")

    if state.get("no_prior_analysis"):
        context = (
            "\nNote: the user seems to be referring to a previous analysis, "
            "but none exists yet in this session. Gently let them know they "
            "need to run an analysis first before you can explain or refine anything."
        )
    else:
        context = ""

    # dataset_summary is normally computed in dataset_node — but
    # advisory_question/general_question skip that node entirely (they go
    # straight from intent_router to direct_answer). So if it's missing
    # here, compute it now, on demand, using the same cache dataset_node
    # itself uses — this way we don't duplicate the inspection work if the
    # summary already exists from an earlier run on this dataset_id.
    dataset_summary = state.get("dataset_summary")
    dataset_id = state.get("dataset_id")

    if dataset_summary is None and dataset_id:
        dataset_summary = dataset_summary_cache.get(dataset_id)
        if dataset_summary is None:
            dataframe = dataset_service.load_dataset(dataset_id)
            dataset_summary = dataset_inspector.inspect(dataframe)
            dataset_summary_cache.set(dataset_id, dataset_summary)
        state["dataset_summary"] = dataset_summary

    if dataset_summary is not None:
        context += (
            "\nHere is the actual dataset the user is asking about — use "
            "these specifics (real column names, types, stats) in your "
            "answer instead of generic advice:\n"
            f"{dataset_summary.model_dump_json(indent=2)}"
        )

    if conversation_id:
        assembled_context = await llm_context_assembler.assemble(
            conversation_id,
            state.get("user_query", ""),
            state.get("dataset_id"),
        )
        context += (
            "\nRelevant assembled context (use only what helps answer the "
            "question):\n"
            f"{assembled_context}"
        )

    messages = [
        {
            "role": "system",
            "content": (
                PIPELINE_SYSTEM_CONTEXT
                + "\n\nAnswer the user's question directly and concisely. "
                "When the user asks about their dataset, ground your answer "
                "in the real columns/stats provided — do not give a generic template."
            ),
        },
        {"role": "user", "content": f"{state.get('user_query', '')}{context}"},
    ]

    response = await llm.ainvoke(messages)
    answer_text = _stringify_llm_content(response.content)
    state["direct_answer"] = answer_text
    if conversation_id:
        await short_term_memory_manager.add_message(
            conversation_id, "assistant", answer_text
        )
    print(
        f"\n========== DIRECT ANSWER ==========\n{answer_text}\n====================================\n")
    return state


async def refine_target_node(state):
    result = await refine_target_agent.identify(state.get("user_query", ""), state.get("llm_config"))

    state["refine_target"] = result.target_stage
    state["refine_instruction"] = result.instruction
    state["refine_confidence"] = result.confidence

    print(f"\n========== REFINE TARGET (Step 1 — not yet confirmed) ==========")
    print(f"Stage: {result.target_stage}")
    print(f"Instruction: {result.instruction}")
    print(f"Confidence: {result.confidence}")
    print(f"Would also rerun: {CASCADE_MAP[result.target_stage]}")
    print("====================================\n")

    return state


def route_intent(state):
    intent = state.get("intent")
    if intent == "run_pipeline":
        # If analysis_plan is already populated, planning happened OUTSIDE
        # the graph (in api/runs.py, before the graph was even chosen —
        # needed so the family/problem_type is known before dispatch).
        # Skip dataset_node/planner_node and go straight to constraint
        # extraction against the already-built plan.
        if state.get("analysis_plan") is not None:
            return "extract_constraints"
        return "run_pipeline"
    if intent == "resume_pipeline":
        if state.get("current_task") or state.get("remaining_tasks"):
            return "resume_pipeline"
        return "run_pipeline"
    if intent == "refine_step":
        return "refine_step"
    return "direct_answer"


async def confirm_refinement_node(state):
    decision = interrupt({
        "type": "refinement",
        "summary": f"Refine '{state.get('refine_target')}': {state.get('refine_instruction')}",
        "stage": state.get("refine_target"),
        "instruction": state.get("refine_instruction"),
        "confidence": state.get("refine_confidence"),
    })

    approved = decision.get("approved", False)
    edit_instruction = decision.get("edit_instruction")

    # An edit is itself a decision to proceed — with the revised
    # instruction — not a rejection. Only a bare Reject (approved=False,
    # no edit_instruction) should actually cancel.
    proceed = approved or bool(edit_instruction)
    state["refine_confirmed"] = proceed

    if proceed:
        if edit_instruction:
            state["refine_instruction"] = edit_instruction

        if _runtime_dataframe(state) is None:
            print("DEBUG: dataframe missing before cascade — reloading from dataset_id")
            dataframe = dataset_service.load_dataset(state["dataset_id"])
            _set_runtime_dataframe(state, dataframe)
            if state.get("dataset_summary") is None:
                state["dataset_summary"] = dataset_summary_cache.get(
                    state["dataset_id"]
                ) or dataset_inspector.inspect(dataframe)
                dataset_summary_cache.set(
                    state["dataset_id"], state["dataset_summary"]
                )

        target = state.get("refine_target")
        refine_instruction = state.get("refine_instruction", "")

        if refine_instruction:
            new_constraints = await constraints_extractor_agent.extract(refine_instruction, state.get("llm_config"))
            existing = dict(state.get("user_constraints") or {})
            for c in new_constraints.constraints:
                existing[c.stage] = [c.instruction]
            state["user_constraints"] = existing

        _clear_stale_fields(state, target)

        state["current_task"] = target
        state["remaining_tasks"] = CASCADE_MAP.get(target, [])
        state["completed_tasks"] = state.get("completed_tasks") or []

    return state


def route_after_confirmation(state):
    return "router" if state.get("refine_confirmed") else "cancelled"


async def refinement_cancelled_node(state):
    print("\nRefinement cancelled by user. No changes made.\n")
    return state


async def extract_constraints_node(state):
    result = await constraints_extractor_agent.extract(state.get("user_query", ""), state.get("llm_config"))

    constraints_by_stage: dict[str, list[str]] = {}
    for c in result.constraints:
        constraints_by_stage.setdefault(c.stage, []).append(c.instruction)

    state["user_constraints"] = constraints_by_stage

    print("\n========== USER CONSTRAINTS (Step 1 — not yet injected) ==========")
    if constraints_by_stage:
        for stage, instructions in constraints_by_stage.items():
            for instr in instructions:
                print(f"  [{stage}] {instr}")
    else:
        print("  (none found)")
    print("====================================\n")

    return state


def _build_plan_summary(plan, constraints: dict) -> str:
    """Human-readable summary of the proposed plan, shown to the user at
    plan_review time. Kept separate from the raw plan object so the
    interrupt payload stays readable rather than dumping a Pydantic repr."""
    if not plan:
        return "No plan generated."

    lines = [
        f"Detected problem type: {plan.problem_type or 'unspecified'}",
        f"Target column: {plan.target_column or 'not detected'}",
        f"Planned stages: {' → '.join(plan.tasks)}",
    ]
    if constraints:
        lines.append("Your explicit instructions:")
        for stage, instrs in constraints.items():
            for instr in instrs:
                lines.append(f"  [{stage}] {instr}")
    return "\n".join(lines)


async def plan_review_node(state):
    # A resumed plan review must not carry a previous direct-answer response.
    state["direct_answer"] = None
    plan = state.get("analysis_plan")
    constraints = state.get("user_constraints") or {}
    dataset_summary = state.get("dataset_summary")

    # Capture the ORIGINAL query once, the first time this node runs —
    # never overwritten again. Every later revision rebuilds from this,
    # instead of stacking edits on top of edits.
    if not state.get("base_user_query"):
        state["base_user_query"] = state.get("user_query")

    validation_problems = validate_plan(plan, dataset_summary)

    decision = interrupt({
        "type": "plan_review",
        "summary": _build_plan_summary(plan, constraints),
        "validation_problems": validation_problems,
        "requires_input": bool(validation_problems),
        "tasks": plan.tasks if plan else [],
        "target_column": state.get("target_column"),
        "problem_type": plan.problem_type if plan else None,
        "constraints": constraints,
    })

    approved = decision.get("approved", False)
    edit_instruction = decision.get("edit_instruction")

    if approved and not edit_instruction:
        state["plan_confirmed"] = True
        state["plan_review_cancelled"] = False
        return state

    if edit_instruction:
        base = state.get("base_user_query") or state.get("user_query", "")

        # Rebuilt fresh from base + only the LATEST edit each time — no
        # stacking, and explicit about which instruction wins if they conflict.
        combined_query = (
            f"{base}\n\n"
            f"IMPORTANT: the user has revised their request. This instruction "
            f"OVERRIDES any conflicting part of the original request above: "
            f"{edit_instruction}"
        )
        state["user_query"] = combined_query

        new_plan = await planner_agent.plan(
            combined_query, state.get(
                "dataset_summary"), state.get("llm_config")
        )

        # Deterministic override: if the user's edit explicitly named a
        # real column as the target, trust that over whatever the LLM
        # inferred from the combined free text — exact-match string
        # logic is more reliable here than hoping the general-purpose
        # planner reliably notices a short instruction buried in a large
        # prompt (this was the actual cause of the plan_review loop).
        explicit_target = extract_explicit_target_column(
            edit_instruction, state.get("dataset_summary")
        )
        if explicit_target:
            new_plan.target_column = explicit_target
            logger.info(
                f"Explicit target column '{explicit_target}' extracted "
                f"from user edit, overriding planner's inference"
            )

        state["analysis_plan"] = new_plan
        state["target_column"] = new_plan.target_column

        new_constraints = await constraints_extractor_agent.extract(combined_query, state.get("llm_config"))
        merged_constraints = dict(constraints)
        for c in new_constraints.constraints:
            merged_constraints[c.stage] = [c.instruction]
        state["user_constraints"] = merged_constraints

        state["plan_confirmed"] = False
        state["plan_review_cancelled"] = False
        return state

    state["plan_confirmed"] = False
    state["plan_review_cancelled"] = True
    return state


def route_after_plan_review(state):
    if state.get("plan_review_cancelled"):
        return "cancelled"
    if state.get("plan_confirmed"):
        return "proceed"
    return "revise"


async def plan_review_cancelled_node(state):
    print("\nPlan rejected by user. No changes made.\n")
    return state

# added to graphs/nodes.py


async def resolve_llm_node(state):
    user_id = state.get("user_id")
    async with async_session() as session:
        llm_config = await resolve_user_llm_config(session, user_id)
    state["llm_config"] = llm_config

    if llm_config:
        logger.info(
            f"[user {user_id}] using BYOK config: "
            f"provider={llm_config.get('provider')} model={llm_config.get('model_name')}"
        )
    else:
        logger.info(
            f"[user {user_id}] no BYOK config — falling back to shared Groq default")

    return state
