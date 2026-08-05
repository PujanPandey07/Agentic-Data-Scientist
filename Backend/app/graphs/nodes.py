from langgraph.graph import END
from agents.planner import planner_agent
from services.dataset_service import dataset_service
from analysis.inspector import dataset_inspector
from services.cleaning_service import CleaningService
from services.executiopn_service import execution_service


def complete_dummy_task(state, task_name: str):

    print(f"Running {task_name} node...")

    execution_service.advance_task(
        state=state,
        completed_task=task_name,
        message=f"{task_name} completed successfully."
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

    state["analysis_plan"] = plan

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

    if task == "training":
        return "training"

    if task == "evaluation":
        return "evaluation"

    if task == "reporting":
        return "reporting"

    return END


async def cleaning_node(state):

    return complete_dummy_task(state, "cleaning")


async def eda_node(state):
    return complete_dummy_task(state, "eda")


async def visualization_node(state):
    return complete_dummy_task(state, "visualization")


async def feature_engineering_node(state):
    return complete_dummy_task(state, "feature_engineering")


async def training_node(state):
    return complete_dummy_task(state, "training")


async def evaluation_node(state):
    return complete_dummy_task(state, "evaluation")


async def reporting_node(state):
    return complete_dummy_task(state, "reporting")
