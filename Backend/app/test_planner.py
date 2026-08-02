import asyncio

from agents.planner import planner_agent
from schema.dataset_summary import DatasetSummary


summary = DatasetSummary(
    rows=1000,
    columns=12,
    column_names=["Age", "Salary", "Purchased"],

    data_types={
        "Age": "int64",
        "Salary": "float64",
        "Purchased": "int64",
    },

    numerical_columns=["Age", "Salary"],
    categorical_columns=[],

    missing_values={},

    duplicate_rows=0,

    memory_usage="120 KB",

    has_missing_values=False,

    has_duplicates=False,

    potential_target_columns=["Purchased"],
)


async def main():
    plan = await planner_agent.plan(
        "Predict whether a customer will purchase a product.",
        summary,
    )

    print(plan)


asyncio.run(main())
