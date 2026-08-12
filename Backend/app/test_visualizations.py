from services.dataset_service import dataset_service
from services.visualization_service import visualization_service
from agents.visualization_planner import visualization_planner_agent


async def main():

    dataset_id = "08f054c7-0d54-4db4-95db-b84616f7bf25"

    print("\n========== LOADING DATASET ==========\n")

    dataframe = dataset_service.load_dataset(dataset_id)

    print(dataframe.head())
    print(f"\nShape: {dataframe.shape}")

    # --------------------------------------------------
    # Create a visualization plan
    # --------------------------------------------------

    visualization_plan = await visualization_planner_agent.plan_visualizations(
        user_query="Analyze the Iris dataset and show useful visualizations.",
        dataset_summary=None,
        eda_report={
            "shape": {
                "rows": len(dataframe),
                "columns": len(dataframe.columns),
            },
            "numerical_columns": [
                "sepal_length",
                "sepal_width",
                "petal_length",
                "petal_width",
            ],
            "categorical_columns": [
                "species"
            ],
        },
    )

    print("\n========== VISUALIZATION PLAN ==========\n")

    print(visualization_plan)

    # --------------------------------------------------
    # Generate charts
    # --------------------------------------------------

    results = visualization_service.generate_visualizations(
        dataframe=dataframe,
        visualization_plan=visualization_plan,
    )

    print("\n========== GENERATED VISUALIZATIONS ==========\n")

    for result in results:
        print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
