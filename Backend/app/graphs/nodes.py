from agents.planner import planner_agent
from services.dataset_service import dataset_service
from analysis.inspector import dataset_inspector


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
