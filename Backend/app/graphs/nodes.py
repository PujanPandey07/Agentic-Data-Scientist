

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


def advance_execution(state, task_name: str, message: str):

    print(f"Running {task_name} node...")

    execution_service.advance_task(
        state=state,
        completed_task=task_name,
        message=message
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
            state, "feature_engineering", "Skipped: missing required inputs"
        )

    service = FeatureEngineeringService()
    transformed_df, report = service.apply_plan(df, plan)

    # The cleaned dataframe is replaced by the engineered dataframe
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
    dataset_id = state.get("dataset_id")  # <-- THIS WAS MISSING

    print(f"DEBUG: df shape = {df.shape if df is not None else None}")
    print(f"DEBUG: plan = {plan is not None}")
    print(f"DEBUG: target_column = {target_column}")
    print(f"DEBUG: dataset_id = {dataset_id}")

    if df is None or plan is None or target_column is None:
        state["training_report"] = {
            "error": "Missing dataframe, model selection plan, or target column"
        }
        return advance_execution(state, "model_selection", "Skipped: missing required inputs")

    service = TrainingService()

    print(f"DEBUG: Calling service.train...")
    final_model, report, best_candidate = service.train(
        df=df,
        plan=plan,
        target_column=target_column,
        dataset_id=dataset_id,  # <-- NOW THIS WORKS
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
    return advance_execution(
        state,
        "evaluation",
        "Model evaluation completed successfully."
    )


async def reporting_node(state):
    return advance_execution(
        state,
        "reporting",
        "Reporting completed successfully."
    )
