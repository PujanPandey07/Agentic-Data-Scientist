import asyncio

from services.dataset_service import dataset_service
from analysis.inspector import dataset_inspector
from services.eda_service import eda_service
from services.cleaning_service import cleaning_service
from agents.visualization_planner import visualization_planner_agent


DATASET_ID = "08f054c7-0d54-4db4-95db-b84616f7bf25"


async def main():

    dataframe = dataset_service.load_dataset(DATASET_ID)

    dataset_summary = dataset_inspector.inspect(dataframe)

    cleaned_dataframe, cleaning_report = (
        cleaning_service.clean_dataset(dataframe)
    )

    eda_report = eda_service.analyze(cleaned_dataframe)

    result = await visualization_planner_agent.plan_visualizations(
        user_query="Analyze the Iris dataset and understand the relationships between the features.",
        dataset_summary=dataset_summary,
        eda_report=eda_report,
    )

    print("\n========== VISUALIZATION PLAN ==========\n")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
