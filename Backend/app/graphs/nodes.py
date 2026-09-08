import logging
from sklearn.model_selection import train_test_split as sk_train_test_split

from utilis.constraints import format_constraints, detect_forced_algorithm
from schema.model_selection import ModelCandidate, ModelSelectionPlan
from langgraph.types import interrupt
from agents.constraint_extractor import constraints_extractor_agent

from agents.refine_target import refine_target_agent
from llm.provider import get_llm
from agents.intent_router import intent_router_agent
from services.hyper_parameters_tuning import HyperparameterTuningService
from services.reporting import ReportingService
from services.evaluation import EvaluationService
from services.trainning import TrainingService
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
from services.cleaning_service import CleaningService
from services.executiopn_service import execution_service
from agents.visualization_planner import visualization_planner_agent
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


def _ensure_train_test_split(state, test_size: float = 0.2, random_state: int = 42):
    """Stratified (classification) or plain (regression) 80/20 train/test
    split, cached in state as train_df/test_df. TrainingService and
    HyperparameterTuningService now only ever see train_df — test_df stays
    untouched until evaluation_node scores the FINAL tuned model against it.

    Cache invalidation: cleaning_node and feature_engineering_node clear
    train_df/test_df whenever they change state["dataframe"], so this
    regenerates automatically after any upstream data change. A refine_step
    that only targets model_selection/hyperparameter_tuning reuses the
    existing split untouched — correct, since the data didn't change.

    Known limitation: feature_engineering_node still fits things like
    scalers/correlation thresholds on the FULL dataframe before this split
    happens, so scaling statistics technically see the test set's
    distribution too. This fixes evaluation leakage (scoring on unseen
    rows), not that subtler feature-engineering-level leakage — a bigger
    restructuring, not addressed here.
    """
    if state.get("train_df") is not None and state.get("test_df") is not None:
        return

    df = state.get("dataframe")
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

    state["train_df"] = train_df.reset_index(drop=True)
    state["test_df"] = test_df.reset_index(drop=True)

    logger.info(
        f"Train/test split created: train={state['train_df'].shape}, "
        f"test={state['test_df'].shape}"
    )


async def dataset_node(state):

    dataframe = dataset_service.load_dataset(
        state["dataset_id"]
    )

    summary = dataset_inspector.inspect(
        dataframe
    )

    state["dataframe"] = dataframe
    state["dataset_summary"] = summary

    return state


async def planner_node(state):

    plan = await planner_agent.plan(
        state["user_query"],
        state["dataset_summary"],
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
    # Record completion from a just-finished viz/FE branch (fan-out or
    # standalone) — done here, in one place, since router only ever
    # runs one instance at a time (no concurrency risk for this update).
    new_completed = []
    new_logs = []

    if state.get("_viz_just_completed"):
        new_completed.append("visualization")
        new_logs.append({"node": "visualization", "status": "success",
                        "message": state["_viz_just_completed"]})
        state["_viz_just_completed"] = None

    if state.get("_fe_just_completed"):
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
        new_remaining = [t for t in remaining if t not in (
            "visualization", "feature_engineering")]
        state["current_task"] = new_remaining[0] if new_remaining else None
        state["remaining_tasks"] = new_remaining[1:] if len(
            new_remaining) > 1 else []
        state["fan_out_viz_fe"] = True
    elif current_task in ("visualization", "feature_engineering"):
        # standalone (non-fanout) case — advance past just this one, as before
        state["current_task"] = remaining[0] if remaining else None
        state["remaining_tasks"] = remaining[1:] if len(remaining) > 1 else []
        state["fan_out_viz_fe"] = False
    else:
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

    dataframe = state.get("dataframe")

    if dataframe is None:
        raise ValueError("Dataframe not found in graph state.")

    cleaned_dataframe, report = cleaning_service.clean_dataset(
        dataframe
    )

    state["dataframe"] = cleaned_dataframe
    state["cleaning_report"] = report
    # Data changed — any cached train/test split is now stale.
    state["train_df"] = None
    state["test_df"] = None

    print("REPORT IN STATE:", state.get("cleaning_report"))

    return advance_execution(
        state,
        "cleaning",
        "Dataset cleaning completed successfully."
    )


async def eda_node(state):
    dataframe = state.get("dataframe")
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
    dataframe = state.get("dataframe")
    if dataframe is None:
        raise ValueError("Dataframe not found in graph state.")

    visualization_plan = state.get("visualization_plan")
    if visualization_plan is None:
        raise ValueError("Visualization plan not found in graph state.")

    visualizations = visualization_service.generate_visualizations(
        dataframe=dataframe,
        visualization_plan=visualization_plan,
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
    )

    return {"feature_engineering_plan": plan}


async def feature_engineering_node(state):
    df = state.get("dataframe")
    plan = state.get("feature_engineering_plan")
    target_column = state.get("target_column")

    print("Running feature_engineering node...")

    if df is None or plan is None:
        return {
            "feature_engineering_report": {
                "error": "Missing dataframe or feature engineering plan"
            },
        }

    # IMPORTANT:
    # Split BEFORE feature engineering so that transformations
    # such as scaling, encoding, variance selection, correlation
    # selection, binning, and polynomial features learn only from train.
    split_state = dict(state)

    _ensure_train_test_split(split_state)

    train_df = split_state.get("train_df")
    test_df = split_state.get("test_df")

    if train_df is None or test_df is None:
        return {
            "feature_engineering_report": {
                "error": "Failed to create train/test split before feature engineering"
            },
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

    return {
        "dataframe": transformed_df,
        "train_df": train_transformed,
        "test_df": test_transformed,
        "feature_engineering_report": report,
        "_fe_just_completed": msg,
    }


async def training_node(state):
    """Deterministic execution: trains models, picks best, saves to disk.

    Trains only on train_df (never test_df) — see _ensure_train_test_split.
    """
    _ensure_train_test_split(state)

    df = state.get("train_df")
    plan = state.get("model_selection_plan")
    target_column = state.get("target_column")
    dataset_id = state.get("dataset_id")

    print(f"DEBUG: df shape = {df.shape if df is not None else None}")
    print(f"DEBUG: plan = {plan is not None}")
    print(f"DEBUG: target_column = {target_column}")
    print(f"DEBUG: dataset_id = {dataset_id}")

    if df is None or plan is None or target_column is None:
        state["training_report"] = {
            "error": "Missing dataframe, model selection plan, or target column"
        }
        return advance_execution(
            state, "model_selection", "Skipped: missing required inputs",
            status="skipped",
        )

    service = TrainingService()

    print(f"DEBUG: Calling service.train...")
    final_model, report, best_candidate = service.train(
        df=df,
        plan=plan,
        target_column=target_column,
        dataset_id=dataset_id,
    )
    print(f"DEBUG: Training done! Best: {best_candidate.algorithm}")

    state["training_report"] = report
    state["trained_model_path"] = report.get("model_path")

    msg = (
        f"Training complete. Best: {best_candidate.algorithm} "
        f"with CV score {report['best_mean_cv_score']}. "
        f"Model saved to {report.get('model_path', 'N/A')}"
    )
    return advance_execution(state, "model_selection", msg)


async def evaluation_node(state):
    """Deterministic evaluation: loads the TUNED model (falling back to the
    untuned trained model if tuning was skipped/failed), scores it against
    the held-out TEST split only — never training data."""
    df = state.get("test_df")
    target_column = state.get("target_column")
    model_path = state.get("tuned_model_path") or state.get(
        "trained_model_path")
    analysis_plan = state.get("analysis_plan")

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

    msg = (
        f"Reporting complete. "
        f"Report saved to {report.get('report_path', 'N/A')}. "
        f"Best model: {report['conclusions']['best_model']} "
        f"(accuracy: {report['conclusions'].get('final_accuracy', 'N/A')})"
    )
    return advance_execution(state, "reporting", msg)


async def hyperparameter_tuning_node(state):
    """Tunes only on train_df (never test_df) — see _ensure_train_test_split."""
    _ensure_train_test_split(state)

    df = state.get("train_df")
    plan = state.get("model_selection_plan")
    training_report = state.get("training_report")
    target_column = state.get("target_column")
    dataset_id = state.get("dataset_id")
    analysis_plan = state.get("analysis_plan")

    if df is None or plan is None or training_report is None or target_column is None or dataset_id is None:
        state["hyperparameter_tuning_report"] = {
            "status": "skipped", "reason": "missing inputs"}
        state["tuned_model_path"] = state.get("trained_model_path")
        return advance_execution(
            state, "hyperparameter_tuning", "Skipped: missing inputs",
            status="skipped",
        )

    # Skip for quick strategy
    if plan.strategy == "quick":
        state["hyperparameter_tuning_report"] = {
            "status": "skipped", "reason": "quick strategy"}
        state["tuned_model_path"] = state.get("trained_model_path")
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
        return advance_execution(
            state, "hyperparameter_tuning", "Skipped: candidate not found",
            status="skipped",
        )

    # analysis_plan may legitimately be None on a thread that's only ever
    # been driven via refine_step (never went through planner_node).
    # Default rather than assume, same as evaluation_node does.
    problem_type = analysis_plan.problem_type if analysis_plan else "classification"

    service = HyperparameterTuningService()
    tuned_model, report = service.tune(
        df=df,
        target_column=target_column,
        best_candidate=best_candidate,
        problem_type=problem_type,
        dataset_id=dataset_id,
        max_trials=20 if plan.strategy == "standard" else 50,
        time_budget_seconds=120 if plan.strategy == "standard" else 300,
    )

    state["hyperparameter_tuning_report"] = report
    state["tuned_model_path"] = report["tuned_model_path"]

    top_param = max(report["param_importance"],
                    key=report["param_importance"].get) if report["param_importance"] else "N/A"
    msg = (
        f"Tuning complete. Score: {report['best_trial_score']}. "
        f"Trials: {report['num_trials_completed']}. "
        f"Top param: {top_param}"
    )
    return advance_execution(state, "hyperparameter_tuning", msg)


async def intent_router_node(state):
    has_prior_report = state.get("final_report") is not None

    classification = await intent_router_agent.classify(
        user_query=state.get("user_query", ""),
        has_prior_report=has_prior_report,
    )

    intent = classification.intent

    if intent in ("explain_result", "refine_step") and not has_prior_report:
        logger.warning(
            f"LLM returned '{intent}' with no prior report — overriding to general_question"
        )
        intent = "general_question"
        state["no_prior_analysis"] = True

    state["intent"] = intent
    return state


async def direct_answer_node(state):
    llm = get_llm()

    if state.get("no_prior_analysis"):
        context = (
            "\nNote: the user seems to be referring to a previous analysis, "
            "but none exists yet in this session. Gently let them know they "
            "need to run an analysis first before you can explain or refine anything."
        )
    elif state.get("final_report"):
        context = f"\nPrevious analysis report: {state['final_report']}"
    else:
        context = ""

    messages = [
        {"role": "system", "content": "Answer the user's question directly and concisely."},
        {"role": "user", "content": f"{state.get('user_query', '')}{context}"},
    ]

    response = await llm.ainvoke(messages)
    state["direct_answer"] = response.content
    print(
        f"\n========== DIRECT ANSWER ==========\n{response.content}\n====================================\n")
    return state


async def refine_target_node(state):
    result = await refine_target_agent.identify(state.get("user_query", ""))

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
        return "run_pipeline"
    if intent == "refine_step":
        return "refine_step"
    return "direct_answer"


async def confirm_refinement_node(state):
    decision = interrupt({
        "question": "Proceed with this refinement?",
        "stage": state.get("refine_target"),
        "instruction": state.get("refine_instruction"),
        "confidence": state.get("refine_confidence"),
    })
    state["refine_confirmed"] = decision

    if decision:
        # ... existing dataframe reload logic stays exactly as-is ...
        if state.get("dataframe") is None:
            print("DEBUG: dataframe missing before cascade — reloading from dataset_id")
            dataframe = dataset_service.load_dataset(state["dataset_id"])
            state["dataframe"] = dataframe
            if state.get("dataset_summary") is None:
                state["dataset_summary"] = dataset_inspector.inspect(dataframe)

        target = state.get("refine_target")

        # Clear the target stage and everything downstream so old results
        # (e.g. a previous training_report) don't mix with new ones.
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
    result = await constraints_extractor_agent.extract(state.get("user_query", ""))

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
