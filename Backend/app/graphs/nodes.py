from services.llm_config import resolve_user_llm_config
from core.db import async_session
from services.job_queue import enqueue_training_job, check_job_result
import logging
from sklearn.model_selection import train_test_split as sk_train_test_split
from services.clustering_evaluation import clustering_evaluation_service
from services.visualization_service import visualization_service
from agents.clustering import clustering_model_selection_planner_agent
from services.job_queue import enqueue_clustering_job
from agents.time_series import time_series_agent
from services.time_series_service import time_series_service

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
from services.job_queue import enqueue_clustering_tuning_job

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
- The user's message becomes a query that a planner turns into a plan: a detected problem_type (classification, regression, clustering, or time_series), an optional target_column, and an ordered list of pipeline stages to run. The user reviews and can approve or edit this plan before anything runs.
- THREE pipeline families are supported, and the system automatically routes to the right one based on the user's request:
  1. Supervised (classification / regression): predict a named target column — e.g. "predict species", "estimate house price". Requires a target_column.
  2. Unsupervised (clustering): find natural groupings, segments, or structure in the data WITHOUT a labeled target — e.g. "segment my customers", "find natural groupings", "detect anomalies". No target_column needed.
  3. Time series: forecast a numeric variable over time — e.g. "forecast next month's sales", "predict future demand". Requires a time column and a target column.
- The stages, always run in this order, are: cleaning -> eda -> visualization -> feature_engineering -> model_selection -> hyperparameter_tuning -> evaluation -> reporting.
- The user can embed per-stage constraints directly in their prompt, e.g. "for feature engineering, one-hot encode the categorical columns", "for model selection, use XGBoost", or "skip visualization" — these are extracted automatically and applied at the matching stage.
- Model selection and hyperparameter tuning run as background jobs; the user sees a "running in the background" status while these complete, rather than the chat blocking.
- After a full run completes, the user gets a final report with conclusions, plus charts and an exportable PDF.
- The user can ask to REFINE a specific completed stage afterward (e.g. "try a different algorithm", "add a chart of X") — this reruns just that stage and everything downstream of it, not the whole pipeline from scratch.
- If a run is interrupted partway (a crash, or the user leaving mid-run), the user can say "continue" or "retry" to resume from where it left off, rather than starting over.
- The user does NOT need to write code or specify implementation details — the system's own deterministic services execute each stage. What matters is that the problem type is clear (or inferable), and that any real preferences are stated explicitly.
- Users can optionally add their own API key (OpenAI, Anthropic, or Gemini) in settings so their runs use their own model instead of the shared default — this only affects which LLM plans/reasons about the data, not the underlying pipeline mechanics.

When (and only when) the user is asking for the best/ideal prompt for their dataset:
1. Inspect the real dataset summary given to you (column names, types, stats) to determine the most likely problem type:
   - If there is a clear categorical or numeric target column the user wants to predict -> classification or regression.
   - If the data has no obvious target and the user would benefit from grouping/segmenting rows -> clustering.
   - If there is a time/date column and a numeric value to forecast -> time_series.
   If it's genuinely ambiguous, say so briefly and ask the user to confirm rather than guessing silently.
2. Produce an actual example prompt, clearly set off (e.g. in a quoted block), that the user could copy and paste directly into this chat to start a run. It should explicitly name:
   - The problem type (classification, regression, clustering, or time_series)
   - The target column (for supervised / time series runs) OR a clear statement that no target is needed (for clustering)
   - Any stages worth emphasizing, skipping, or constraining, only if there's a real reason based on THIS dataset — don't pad it with generic advice that applies to every dataset.
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
    "model_selection": [
        "model_selection_plan", "training_report", "trained_model_path",
        "clustering_model_selection_plan", "clustering_training_report", "clustering_model_path",
    ],
    "hyperparameter_tuning": [
        "hyperparameter_tuning_report", "tuned_model_path",
        "clustering_tuning_report", "tuned_clustering_model_path",
        "cluster_labels", "clustering_elbow_curve", "clustering_linkage_matrix",
    ],
    "evaluation": ["evaluation_report", "clustering_evaluation_report"],
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

    Clustering has no labels to stratify or generalize-test against, so it
    intentionally skips splitting entirely and fits on the full dataset —
    this is checked explicitly by problem_type rather than relying on
    target_column being None as an implicit signal, so the behavior stays
    correct even if that assumption ever changes upstream.
    """
    analysis_plan = state.get("analysis_plan")
    problem_type = analysis_plan.problem_type if analysis_plan else "classification"

    if problem_type == "clustering":
        return

    train_df, test_df = _runtime_train_test(state)
    if train_df is not None and test_df is not None:
        return

    df = _runtime_dataframe(state)
    target_column = state.get("target_column")

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
        # Persist updated (snake_case) summary back to cache so follow-up
        # messages see correct post-cleaning column names instead of stale
        # original-cased names.  Without this update, plan_review, the
        # planner, and direct_answer_node all receive outdated column lists.
        dataset_id = state.get("dataset_id")
        if dataset_id:
            dataset_summary_cache.set(dataset_id, summary)
            logger.info(
                "dataset_summary_cache updated with post-cleaning column names "
                "for dataset_id=%s", dataset_id
            )

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
    analysis_plan = state.get("analysis_plan")
    problem_type = analysis_plan.problem_type if analysis_plan else None

    if problem_type == "clustering":
        return await _feature_engineering_clustering_node(state)

    if problem_type == "time_series":
        # Time-series feature engineering is lag/window based, performed directly in enqueue_ts_node
        return advance_execution(
            state, "feature_engineering",
            "Time-series temporal features (lags and rolling windows) reserved for sequential pipeline."
        )

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

    # ZERO-FEATURE GUARD: Ensure candidate models have predictor columns
    feature_cols = [c for c in train_transformed.columns if c != target_column]
    if not feature_cols:
        logger.warning("Feature engineering removed all predictor columns. Falling back to original clean numeric columns.")
        num_cols = train_df.select_dtypes(include="number").columns
        fallback_cols = [c for c in num_cols if c != target_column]
        if fallback_cols:
            train_transformed[fallback_cols] = train_df[fallback_cols]
            test_transformed[fallback_cols] = test_df[fallback_cols]

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


async def _feature_engineering_clustering_node(state):
    """Clustering variant: no target column, no train/test split — fits
    transformations on and applies them to the ENTIRE dataset via
    FeatureEngineeringService.apply_plan (the single-dataframe method that
    already existed for exactly this shape of use, alongside
    apply_plan_train_test). This is where a dimensionality-reduction step
    like PCA would typically be applied, if the plan includes one, before
    the clustering algorithm itself ever sees the data.
    """
    df = _runtime_dataframe(state)
    plan = state.get("feature_engineering_plan")

    print("Running feature_engineering node (clustering)...")

    if df is None or plan is None:
        return {
            "feature_engineering_report": {
                "error": "Missing dataframe or feature engineering plan"
            },
            "_fe_just_completed": "Skipped: missing dataframe or feature engineering plan",
        }

    service = FeatureEngineeringService()
    transformed_df, report = service.apply_plan(
        df,
        plan,
        target_column=None,
    )

    msg = (
        f"Feature engineering complete. "
        f"{df.shape} -> {transformed_df.shape}. "
        f"Steps: {report['steps_executed']}/{len(plan.steps)} succeeded."
    )

    _set_runtime_dataframe(state, transformed_df)
    # No train/test split exists for clustering — make sure neither key is
    # left stale from an earlier stage or a prior run on this thread.
    state["train_df"] = None
    state["test_df"] = None

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


async def enqueue_clustering_tuning_node(state):
    dataset_id = state.get("dataset_id")
    training_report = state.get("clustering_training_report")

    if training_report is None or "best_algorithm" not in training_report:
        state["clustering_tuning_report"] = {
            "status": "skipped", "reason": "missing inputs"}
        state["tuned_clustering_model_path"] = state.get(
            "clustering_model_path")
        state["_clustering_tuning_job_id"] = None
        return advance_execution(
            state, "hyperparameter_tuning", "Skipped: missing inputs",
            status="skipped",
        )

    plan_dict = state.get("clustering_model_selection_plan") or {}
    scoring_metric = plan_dict.get("scoring_metric", "silhouette")

    best_candidate_dict = {
        "algorithm": training_report["best_algorithm"],
        "reason": training_report.get("best_reason", ""),
        "estimated_time_seconds": 30,
        "hyperparams": training_report.get("best_hyperparams", {}),
        "priority": 1,
    }

    job_id = await enqueue_clustering_tuning_job(
        dataset_id, best_candidate_dict, scoring_metric,
        max_trials=20, time_budget_seconds=120,
    )
    state["_clustering_tuning_job_id"] = job_id
    return state


async def poll_clustering_tuning_node(state):
    job_id = state.get("_clustering_tuning_job_id")

    if job_id is None:
        return state

    outcome = await check_job_result(job_id)

    if outcome is None:
        interrupt({
            "type": "job_status",
            "job_type": "clustering_tuning",
            "job_id": job_id,
            "summary": "Tuning cluster count/parameters in the background — this may take a moment.",
        })
        return state

    state["_clustering_tuning_job_id"] = None

    if outcome["status"] == "failed":
        state["clustering_tuning_report"] = {
            "status": "failed", "error": outcome["error"]}
        state["tuned_clustering_model_path"] = state.get(
            "clustering_model_path")
        return advance_execution(
            state, "hyperparameter_tuning",
            f"Clustering tuning failed: {outcome['error']}",
            status="skipped",
        )

    result = outcome["result"]
    state["clustering_tuning_report"] = result
    state["tuned_clustering_model_path"] = result["tuned_model_path"]
    state["cluster_labels"] = result.get("cluster_labels")
    state["clustering_elbow_curve"] = result.get("elbow_curve")
    state["clustering_linkage_matrix"] = result.get("linkage_matrix")

    # Deterministic clustering charts — generated here (not via the LLM
    # visualization planner) because this is the earliest point in the
    # pipeline where cluster labels/elbow curve/linkage matrix actually
    # exist. Appended into the SAME visualization_results list the
    # earlier (EDA-based) visualization stage already populated, so
    # reporting_node needs no changes to pick these up.
    dataframe = _runtime_dataframe(state)
    if dataframe is not None:
        clustering_charts = visualization_service.generate_clustering_visualizations(
            dataframe=dataframe,
            cluster_labels=state.get("cluster_labels"),
            elbow_curve=state.get("clustering_elbow_curve"),
            linkage_matrix=state.get("clustering_linkage_matrix"),
            dataset_id=state.get("dataset_id"),
        )
        state["visualization_results"] = (
            state.get("visualization_results") or []) + clustering_charts

    msg = (
        f"Tuning complete. {result['n_clusters_found']} clusters found, "
        f"{result['scoring_metric']} score {result['best_trial_score']}. "
        f"Trials: {result['num_trials_completed']}"
    )
    return advance_execution(state, "hyperparameter_tuning", msg)


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

    if df is None:
        logger.info("Test split not found in runtime cache for evaluation; reconstructing split.")
        _ensure_train_test_split(state)
        _, df = _runtime_train_test(state)

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

    # ── Persist previous-run summary BEFORE overwriting the cache ───────────
    # Each time the same dataset is analysed with a different algorithm the
    # old context is lost. We snapshot it into long_term_memory so the
    # chatbot can answer comparison questions like "how did random forest
    # compare to XGBoost last time?".
    conversation_id = state.get("conversation_id") or report.get("dataset_id")
    try:
        existing_ctx = analysis_context_cache.get(report["dataset_id"])
        if existing_ctx is not None and conversation_id:
            prev_conc = existing_ctx.conclusions or {}
            prev_algo = (
                prev_conc.get("best_model")
                or prev_conc.get("best_algorithm")
                or "unknown"
            )
            prev_type = (existing_ctx.dataset or {}).get("problem_type", "unknown")
            prev_target = (existing_ctx.dataset or {}).get("target_column", "")

            # Build a compact human-readable summary line.
            score_parts = []
            for key in ("accuracy", "f1_weighted", "rmse", "r2", "silhouette"):
                val = prev_conc.get(key)
                if val is not None:
                    score_parts.append(f"{key}={val}")
            score_str = ", ".join(score_parts) if score_parts else "scores not recorded"

            summary_line = (
                f"Previous run on this dataset — "
                f"problem type: {prev_type}, "
                f"target column: {prev_target or 'N/A'}, "
                f"best model/algorithm: {prev_algo}, "
                f"scores: {score_str}."
            )

            await long_term_memory_manager.upsert(
                conversation_id,
                "run_history",          # stable topic key — accumulates across runs
                summary_line,
                importance=0.65,
            )
    except Exception:
        logger.warning(
            "Could not store previous-run summary in long_term_memory",
            exc_info=True,
        )

    # Overwrite the cache with the latest run context.
    analysis_context_cache.set(report["dataset_id"], analysis_context)

    recommendation = (report.get("conclusions") or {}).get("recommendation")
    if conversation_id and recommendation and recommendation != "N/A":
        await long_term_memory_manager.upsert(
            conversation_id,
            "latest_analysis_conclusion",
            recommendation,
            importance=0.75,
        )

    # .get() with fallbacks instead of ['best_model']: the shape of
    # "conclusions" differs by pipeline type, and a missing key used to
    # raise KeyError here AFTER the report was built, so LangGraph never
    # committed final_report. Clustering reports have no 'best_model'.
    conclusions = report.get("conclusions") or {}
    best_model = (
        conclusions.get("best_model")
        or conclusions.get("best_algorithm")
        or "N/A"
    )

    msg = (
        f"Reporting complete. "
        f"Report saved to {report.get('report_path', 'N/A')}. "
        f"Best model: {best_model}"
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

    if state.get("intent"):
        return state

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
    dataset_id = state.get("dataset_id")

    # Refresh dataset_summary from cache if present so we always validate against
    # the latest state (e.g. if cleaned columns were standardized)
    if dataset_id and dataset_summary_cache.get(dataset_id):
        dataset_summary = dataset_summary_cache.get(dataset_id)
        state["dataset_summary"] = dataset_summary
    else:
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

        # Use latest dataset_summary for replanning
        current_summary = state.get("dataset_summary") or (dataset_summary_cache.get(dataset_id) if dataset_id else None)
        new_plan = await planner_agent.plan(
            combined_query, current_summary, state.get("llm_config")
        )

        # Deterministic override: if the user's edit explicitly named a
        # real column as the target, trust that over whatever the LLM
        # inferred from the combined free text.
        explicit_target = extract_explicit_target_column(
            edit_instruction, current_summary
        )
        if explicit_target:
            new_plan.target_column = explicit_target
            logger.info(
                f"Explicit target column '{explicit_target}' extracted "
                f"from user edit, overriding planner's inference"
            )

        # Deterministic override for problem_type: if user explicitly specifies
        # switching or using a specific problem type, override it
        edit_lower = edit_instruction.lower()
        if "regression" in edit_lower:
            new_plan.problem_type = "regression"
            if "model_selection" not in (new_plan.tasks or []):
                new_plan.tasks = ["cleaning", "eda", "visualization", "feature_engineering", "model_selection", "hyperparameter_tuning", "evaluation", "reporting"]
        elif "classification" in edit_lower:
            new_plan.problem_type = "classification"
            if "model_selection" not in (new_plan.tasks or []):
                new_plan.tasks = ["cleaning", "eda", "visualization", "feature_engineering", "model_selection", "hyperparameter_tuning", "evaluation", "reporting"]
        elif "cluster" in edit_lower:
            new_plan.problem_type = "clustering"
            new_plan.target_column = None
        elif "time series" in edit_lower or "forecast" in edit_lower:
            new_plan.problem_type = "time_series"

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


async def clustering_evaluation_node(state):
    """Deterministic evaluation: no model fitting, just scores the final
    cluster_labels (produced by tuning, or training if tuning was
    skipped) against the dataset that produced them."""
    df = _runtime_dataframe(state)
    labels = state.get("cluster_labels")
    dataset_id = state.get("dataset_id")

    if df is None or labels is None:
        state["clustering_evaluation_report"] = {
            "error": "Missing dataframe or cluster labels"
        }
        return advance_execution(
            state, "evaluation", "Skipped: missing required inputs",
            status="skipped",
        )

    report = clustering_evaluation_service.evaluate(df, labels, dataset_id)
    state["clustering_evaluation_report"] = report

    if "error" in report:
        return advance_execution(
            state, "evaluation", f"Skipped: {report['error']}",
            status="skipped",
        )

    metrics = report["metrics"]
    if "silhouette_score" in metrics:
        msg = (
            f"Evaluation complete. {report['n_clusters']} clusters, "
            f"silhouette: {metrics['silhouette_score']}, "
            f"Davies-Bouldin: {metrics['davies_bouldin_index']}"
        )
    else:
        msg = f"Evaluation complete with limited metrics: {metrics.get('warning', '')}"

    return advance_execution(state, "evaluation", msg)


def route_task_unsupervised(state):
    """Same task-name strings as route_task (cleaning/eda/feature_engineering/
    model_selection/hyperparameter_tuning/evaluation/reporting) — only the
    NODE each task routes to differs, since the underlying stage semantics
    differ for clustering. Reuses the same fan_out_viz_fe mechanism from
    the shared router() node unchanged."""
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


async def clustering_model_selection_planner_node(state):
    """LLM reasoning: decides WHICH clustering algorithms to try. Mirrors
    model_selection_planner_node's shape, minus the forced-algorithm-only
    fast path — clustering's forced_algorithm handling lives entirely in
    ClusteringTrainingService.train() instead, since the LLM call here is
    cheap and skipping it doesn't save much."""
    constraints = (state.get("user_constraints")
                   or {}).get("model_selection", [])

    plan = await clustering_model_selection_planner_agent.plan(
        user_query=state.get("user_query", ""),
        dataset_summary=state.get("dataset_summary"),
        eda_report=state.get("eda_report", {}),
        feature_engineering_report=state.get("feature_engineering_report"),
        constraints=constraints,
        llm_config=state.get("llm_config"),
    )

    state["clustering_model_selection_plan"] = plan.model_dump()
    return state


async def enqueue_clustering_node(state):
    dataset_id = state.get("dataset_id")
    plan_dict = state.get("clustering_model_selection_plan")

    if plan_dict is None:
        state["clustering_training_report"] = {
            "error": "Missing clustering model selection plan"}
        state["_clustering_job_id"] = None
        return advance_execution(
            state, "model_selection", "Skipped: missing required inputs",
            status="skipped",
        )

    job_id = await enqueue_clustering_job(dataset_id, plan_dict)
    state["_clustering_job_id"] = job_id
    return state


async def poll_clustering_node(state):
    job_id = state.get("_clustering_job_id")

    if job_id is None:
        # enqueue_clustering_node already handled a skip case above and
        # advanced past model_selection itself — nothing to poll.
        return state

    outcome = await check_job_result(job_id)

    if outcome is None:
        interrupt({
            "type": "job_status",
            "job_type": "clustering",
            "job_id": job_id,
            "summary": "Clustering is running in the background — this may take a moment.",
        })
        return state

    state["_clustering_job_id"] = None

    if outcome["status"] == "failed":
        state["clustering_training_report"] = {"error": outcome["error"]}
        return advance_execution(
            state, "model_selection",
            f"Clustering failed: {outcome['error']}",
            status="skipped",
        )

    result = outcome["result"]
    state["clustering_training_report"] = result
    state["clustering_model_path"] = result.get("model_path")
    state["cluster_labels"] = result.get("cluster_labels")

    msg = (
        f"Clustering complete. Best: {result['best_algorithm']} "
        f"({result['n_clusters_found']} clusters found, "
        f"{result['scoring_metric']} score {result['best_score']}). "
        f"Model saved to {result.get('model_path', 'N/A')}"
    )
    return advance_execution(state, "model_selection", msg)


# ════════════════════════════════════════════════════════════════════════════
# TIME SERIES NODES
# ════════════════════════════════════════════════════════════════════════════

async def ts_analysis_node(state):
    """
    Stationarity test + seasonal decomposition.
    Produces ts_plan (from LLM) and ts_analysis_report (from maths).
    """
    dataset_id = state.get("dataset_id")
    df = runtime_state_store.get_dataset(dataset_id)

    if df is None:
        return advance_execution(
            state, "ts_analysis",
            "Skipped: dataset not available in runtime cache.",
            status="skipped",
        )

    dataset_summary = state.get("dataset_summary")
    llm_config = state.get("llm_config")
    user_query = state.get("user_query", "")

    # 1. LLM produces the plan
    try:
        plan = await time_series_agent.analyze(user_query, dataset_summary, llm_config)
    except Exception as exc:
        logger.warning("TS agent analyze failed: %s", exc)
        # Detect time column automatically as fallback
        time_col = time_series_service.detect_time_column(df)
        from schema.time_series_plan import TimeSeriesPlan
        plan = TimeSeriesPlan(
            time_column=time_col or df.columns[0],
            target_column=state.get("target_column") or df.select_dtypes(
                "number").columns[-1],
        )

    # Validate columns exist in df
    if plan.time_column not in df.columns:
        detected = time_series_service.detect_time_column(df)
        if detected:
            plan.time_column = detected
        elif len(df.columns) > 0:
            plan.time_column = df.columns[0]

    # Resolve target column case/presence
    if plan.target_column not in df.columns:
        # Try case-insensitive / normalized lookup
        norm_map = {c.lower().replace(" ", "_"): c for c in df.columns}
        target_norm = (plan.target_column or "").lower().replace(" ", "_")
        if target_norm in norm_map:
            plan.target_column = norm_map[target_norm]
        else:
            nums = df.select_dtypes("number").columns.tolist()
            if nums:
                plan.target_column = nums[-1]
            else:
                other_cols = [c for c in df.columns if c != plan.time_column]
                plan.target_column = other_cols[-1] if other_cols else (df.columns[0] if len(df.columns) > 0 else "target")

    # 2. Parse datetime column
    try:
        if plan.time_column in df.columns:
            df[plan.time_column] = pd.to_datetime(
                df[plan.time_column], errors="coerce"
            )
            df = df.sort_values(plan.time_column).reset_index(drop=True)
            runtime_state_store.set_dataset(dataset_id, df)
    except Exception as exc:
        logger.warning("Could not parse time column: %s", exc)

    # 3. Stationarity test
    if plan.target_column in df.columns:
        # Ensure series is numeric
        df[plan.target_column] = pd.to_numeric(df[plan.target_column], errors="coerce")
        runtime_state_store.set_dataset(dataset_id, df)
        series = df[plan.target_column].dropna()
    else:
        series = pd.Series(dtype=float)
    _, is_stationary, n_diffs = time_series_service.make_stationary(series)
    plan.is_stationary = is_stationary

    # 4. Decomposition
    decomp = {}
    if plan.time_column in df.columns and plan.target_column in df.columns:
        decomp = time_series_service.decompose(
            df, plan.time_column, plan.target_column, plan.frequency
        )
    plan.seasonality_detected = decomp.get("seasonality_detected", False)
    plan.trend_detected = decomp.get("trend_detected", False)

    ts_analysis_report = {
        "time_column": plan.time_column,
        "target_column": plan.target_column,
        "n_rows": int(len(df)),
        "frequency": plan.frequency,
        "is_stationary": is_stationary,
        "n_diffs_applied": n_diffs,
        "trend_detected": plan.trend_detected,
        "seasonality_detected": plan.seasonality_detected,
        "decomposition": {
            "trend_sample": decomp.get("trend", [])[:20],
            "period": decomp.get("period"),
        },
    }

    state["ts_plan"] = plan.model_dump()
    state["ts_analysis_report"] = ts_analysis_report
    # Propagate target_column into main state so shared nodes (e.g. cleaning) see it
    state["target_column"] = plan.target_column

    msg = (
        f"Time-series analysis complete. "
        f"time_col={plan.time_column}, target={plan.target_column}, "
        f"stationary={is_stationary}, trend={plan.trend_detected}, "
        f"seasonality={plan.seasonality_detected}"
    )
    return advance_execution(state, "ts_analysis", msg)


async def ts_model_selection_planner_node(state):
    """LLM picks the best lag config and model for forecasting."""
    # Ensure ts_analysis has executed and produced ts_plan and ts_analysis_report
    if state.get("ts_plan") is None:
        logger.info("ts_plan missing in ts_model_selection_planner_node; executing ts_analysis_node inline first.")
        await ts_analysis_node(state)

    ts_plan_dict = state.get("ts_plan")
    ts_analysis_report = state.get("ts_analysis_report") or {}

    if ts_plan_dict is None:
        state["ts_model_selection_plan"] = {
            "primary_model": "xgboost_lags",
            "lag_features": [1, 2, 3, 7],
            "rolling_windows": [3, 7],
            "reasoning": "Default — ts_plan missing.",
        }
        return state

    from schema.time_series_plan import TimeSeriesPlan
    ts_plan = TimeSeriesPlan.model_validate(ts_plan_dict)

    try:
        plan = await time_series_agent.select_model(
            ts_plan, ts_analysis_report, state.get("llm_config")
        )
    except Exception as exc:
        logger.warning("TS model selection failed: %s", exc)
        plan = {
            "primary_model": "xgboost_lags",
            "lag_features": ts_plan.lag_features,
            "rolling_windows": ts_plan.rolling_windows,
            "reasoning": "Fallback due to error.",
        }

    state["ts_model_selection_plan"] = plan
    return state


async def enqueue_ts_node(state):
    """Prepare and run the time-series training synchronously (no background job needed — fast)."""
    dataset_id = state.get("dataset_id")
    ts_plan_dict = state.get("ts_plan")
    ts_model_plan = state.get("ts_model_selection_plan") or {}

    df = runtime_state_store.get_dataset(dataset_id)

    if df is None or ts_plan_dict is None:
        state["ts_training_report"] = {"error": "Missing dataset or ts_plan"}
        state["_ts_job_id"] = None
        return advance_execution(
            state, "model_selection", "Skipped: missing inputs", status="skipped"
        )

    from schema.time_series_plan import TimeSeriesPlan
    ts_plan = TimeSeriesPlan.model_validate(ts_plan_dict)

    lags = ts_model_plan.get("lag_features", ts_plan.lag_features)
    rolling = ts_model_plan.get("rolling_windows", ts_plan.rolling_windows)

    # Create lag features
    try:
        featured_df = time_series_service.create_lag_features(
            df, ts_plan.target_column, lags, rolling, ts_plan.time_column
        )
    except Exception as exc:
        state["ts_training_report"] = {
            "error": f"Feature engineering failed: {exc}"}
        state["_ts_job_id"] = None
        return advance_execution(state, "model_selection", f"TS feature error: {exc}", status="skipped")

    # Chronological split
    train_df, test_df = time_series_service.chronological_split(
        featured_df, test_ratio=0.20, time_col=ts_plan.time_column
    )

    # Store splits for evaluation
    runtime_state_store.set_dataset(f"{dataset_id}_ts_train", train_df)
    runtime_state_store.set_dataset(f"{dataset_id}_ts_test", test_df)

    feature_cols = time_series_service.get_feature_columns(
        featured_df, ts_plan.target_column, ts_plan.time_column
    )

    if not feature_cols:
        state["ts_training_report"] = {
            "error": "No lag features generated — too few rows?"}
        state["_ts_job_id"] = None
        return advance_execution(state, "model_selection", "No lag features", status="skipped")

    # Fit model (synchronous — fast enough for forecasting)
    result = time_series_service.fit_xgboost_lags(
        train_df, test_df, ts_plan.target_column, feature_cols
    )
    result["feature_cols"] = feature_cols
    result["lag_features_used"] = lags
    result["rolling_windows_used"] = rolling

    state["ts_training_report"] = result
    state["_ts_job_id"] = "done"  # no actual background job

    msg = (
        f"TS training complete ({result.get('model', 'XGBoost')}): "
        f"MAE={result.get('metrics', {}).get('mae')}, "
        f"RMSE={result.get('metrics', {}).get('rmse')}, "
        f"MAPE={result.get('metrics', {}).get('mape')}%"
    )
    return advance_execution(state, "model_selection", msg)


async def poll_ts_node(state):
    """TS training is synchronous — just pass through."""
    # Training is done inline in enqueue_ts_node, so poll is always immediate
    return state


async def ts_evaluation_node(state):
    """Holdout evaluation on the test set — computes final forecast metrics."""
    ts_plan_dict = state.get("ts_plan")
    ts_training = state.get("ts_training_report") or {}
    dataset_id = state.get("dataset_id")

    if "error" in ts_training or not ts_plan_dict:
        state["ts_evaluation_report"] = {
            "error": "Training did not complete successfully"}
        return advance_execution(state, "evaluation", "TS evaluation skipped", status="skipped")

    from schema.time_series_plan import TimeSeriesPlan
    ts_plan = TimeSeriesPlan.model_validate(ts_plan_dict)

    metrics = ts_training.get("metrics", {})
    n_test = ts_training.get("n_test", 0)

    evaluation_report = {
        "problem_type": "time_series",
        "model": ts_training.get("model", "XGBoost"),
        "n_test_samples": n_test,
        "target_column": ts_plan.target_column,
        "time_column": ts_plan.time_column,
        "metrics": {
            "mae":  metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "mape": metrics.get("mape"),
        },
        "feature_importances": ts_training.get("feature_importances", {}),
        "predictions_sample": ts_training.get("predictions", [])[:20],
        "actuals_sample": ts_training.get("actuals", [])[:20],
    }

    # MAPE-based quality label
    mape = metrics.get("mape")
    if mape is not None:
        if mape < 5:
            evaluation_report["quality"] = "excellent"
        elif mape < 10:
            evaluation_report["quality"] = "good"
        elif mape < 20:
            evaluation_report["quality"] = "fair"
        else:
            evaluation_report["quality"] = "poor"

    state["ts_evaluation_report"] = evaluation_report
    msg = f"TS evaluation: MAE={metrics.get('mae')}, RMSE={metrics.get('rmse')}, MAPE={metrics.get('mape')}%"
    return advance_execution(state, "evaluation", msg)


def route_task_ts(state) -> str:
    """
    Router for the time-series graph.

    Returns task-name strings that EXACTLY match the keys in
    build_time_series_graph()'s add_conditional_edges path map.
    """
    current_task = state.get("current_task")
    remaining = state.get("remaining_tasks", [])

    if not current_task and not remaining:
        return END

    task = current_task or (remaining[0] if remaining else None)
    if task is None:
        return END

    # Map plan task names → path-map keys
    _MAP = {
        "cleaning":              "cleaning",
        "eda":                   "eda",
        "ts_analysis":           "ts_analysis",
        "visualization":         "visualization",
        "feature_engineering":   "feature_engineering",
        "model_selection":       "model_selection",
        "hyperparameter_tuning": "hyperparameter_tuning",
        "evaluation":            "evaluation",
        "reporting":             "reporting",
    }
    return _MAP.get(task, END)
