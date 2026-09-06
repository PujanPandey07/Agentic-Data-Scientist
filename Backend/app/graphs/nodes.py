from langgraph.types import interrupt
from fastapi import logger

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

    return state


async def router(state):
    return state


def route_task(state):

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
        eda_report=eda_report
    )

    state["visualization_plan"] = visualization_plan

    return state


async def visualization_node(state):

    dataframe = state.get("dataframe")

    if dataframe is None:
        raise ValueError(
            "Dataframe not found in graph state."
        )

    visualization_plan = state.get("visualization_plan")

    if visualization_plan is None:
        raise ValueError(
            "Visualization plan not found in graph state."
        )

    visualizations = visualization_service.generate_visualizations(
        dataframe=dataframe,
        visualization_plan=visualization_plan,
    )

    state["visualizations"] = visualizations

    return advance_execution(
        state,
        "visualization",
        "Visualizations generated successfully."
    )


async def feature_engineering_planner_node(state):
    """LLM reasoning: decides WHAT feature engineering to do."""
    agent = FeatureEngineeringPlannerAgent()

    plan = await agent.plan(
        user_query=state.get("user_query", ""),
        dataset_summary=state.get("dataset_summary"),
        eda_report=state.get("eda_report", {}),
    )

    state["feature_engineering_plan"] = plan
    return state


async def feature_engineering_node(state):
    """Deterministic execution: applies the plan to the dataframe."""
    df = state.get("dataframe")
    plan = state.get("feature_engineering_plan")

    if df is None or plan is None:
        state["feature_engineering_report"] = {
            "error": "Missing dataframe or feature engineering plan"
        }
        return advance_execution(
            state, "feature_engineering", "Skipped: missing required inputs",
            status="skipped",
        )

    service = FeatureEngineeringService()
    transformed_df, report = service.apply_plan(df, plan)

    state["dataframe"] = transformed_df
    state["feature_engineering_report"] = report

    msg = (
        f"Feature engineering complete. "
        f"Shape: {df.shape} -> {transformed_df.shape}. "
        f"Steps: {report['steps_executed']}/{len(plan.steps)} succeeded."
    )
    return advance_execution(state, "feature_engineering", msg)


async def model_selection_planner_node(state):
    """LLM reasoning: decides WHICH models to try."""
    agent = ModelSelectionPlannerAgent()

    plan = await agent.plan(
        user_query=state.get("user_query", ""),
        dataset_summary=state.get("dataset_summary"),
        eda_report=state.get("eda_report", {}),
        feature_engineering_report=state.get("feature_engineering_report"),
    )

    state["model_selection_plan"] = plan
    return state


async def training_node(state):
    """Deterministic execution: trains models, picks best, saves to disk."""
    df = state.get("dataframe")
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
    """Deterministic evaluation: loads model, computes metrics, generates artifacts."""
    df = state.get("dataframe")
    target_column = state.get("target_column")
    model_path = state.get("trained_model_path")
    analysis_plan = state.get("analysis_plan")

    if df is None or target_column is None or model_path is None:
        state["evaluation_report"] = {
            "error": "Missing dataframe, target column, or trained model path"
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
    df = state.get("dataframe")
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
        # refine_step never routes through dataset_node, so a thread that's
        # only ever been driven via refine_step has no dataframe in its
        # checkpoint yet. Reload it here rather than silently cascading
        # through every node's "missing input" skip branch.
        if state.get("dataframe") is None:
            print("DEBUG: dataframe missing before cascade — reloading from dataset_id")
            dataframe = dataset_service.load_dataset(state["dataset_id"])
            state["dataframe"] = dataframe
            if state.get("dataset_summary") is None:
                state["dataset_summary"] = dataset_inspector.inspect(dataframe)

        target = state.get("refine_target")
        state["current_task"] = target
        state["remaining_tasks"] = CASCADE_MAP.get(target, [])
        state["completed_tasks"] = state.get("completed_tasks") or []

    return state


def route_after_confirmation(state):
    return "router" if state.get("refine_confirmed") else "cancelled"


async def refinement_cancelled_node(state):
    print("\nRefinement cancelled by user. No changes made.\n")
    return state
