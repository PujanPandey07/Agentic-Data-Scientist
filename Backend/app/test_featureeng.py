import pandas as pd
import asyncio

from services.feature_engineering import FeatureEngineeringService
from agents.feature_engineering_planner import FeatureEngineeringPlannerAgent


async def main():
    print("\n========== MOCK DATASET ==========\n")

    # Create Iris-like mock data (150 rows, 5 columns)
    # You can replace this with: df = dataset_service.load_dataset(dataset_id)
    df = pd.DataFrame({
        "sepal_length": [5.1, 4.9, 4.7, 5.0, 5.4, 5.7, 5.1, 5.4, 5.1, 4.6] * 15,
        "sepal_width":  [3.5, 3.0, 3.2, 3.6, 3.9, 3.0, 3.4, 3.9, 3.5, 3.4] * 15,
        "petal_length": [1.4, 1.4, 1.3, 1.4, 1.7, 4.2, 4.5, 4.5, 4.0, 1.0] * 15,
        "petal_width":  [0.2, 0.2, 0.2, 0.2, 0.4, 1.2, 1.5, 1.5, 1.3, 0.2] * 15,
        "species":      (["setosa"] * 75) + (["versicolor"] * 45) + (["virginica"] * 30),
    })

    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # --------------------------------------------------
    # Mock EDA report (replace with real eda_service.analyze(df) if available)
    # --------------------------------------------------
    eda_report = {
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "column_names": list(df.columns),
        "numerical_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "categorical_columns": ["species"],
        "numerical_summary": {
            "sepal_length": {"mean": 5.84, "std": 0.83, "skewness": 0.31},
            "sepal_width":  {"mean": 3.05, "std": 0.43, "skewness": 0.32},
            "petal_length": {"mean": 3.76, "std": 1.76, "skewness": -0.27},
            "petal_width":  {"mean": 1.20, "std": 0.76, "skewness": -0.10},
        },
        "categorical_summary": {
            "species": {"unique": 3, "top": "setosa", "freq": 75}
        },
        "missing_values": {},
        "correlation": {
            "sepal_length": {"sepal_width": -0.11, "petal_length": 0.87, "petal_width": 0.82},
            "sepal_width":  {"sepal_length": -0.11, "petal_length": -0.42, "petal_width": -0.36},
            "petal_length": {"sepal_length": 0.87, "sepal_width": -0.42, "petal_width": 0.96},
            "petal_width":  {"sepal_length": 0.82, "sepal_width": -0.36, "petal_length": 0.96},
        },
    }

    user_query = "Build a classification model to predict iris species"

    # --------------------------------------------------
    # 1. Planner Agent: LLM reasons and generates structured plan
    # --------------------------------------------------

    print("\n========== FEATURE ENGINEERING PLANNER ==========\n")

    agent = FeatureEngineeringPlannerAgent()
    plan = await agent.plan(
        user_query=user_query,
        dataset_summary={"columns": list(df.columns), "shape": df.shape},
        eda_report=eda_report,
    )

    print(f"Strategy: {plan.strategy}")
    print(f"Steps: {len(plan.steps)}")
    for i, step in enumerate(plan.steps, 1):
        print(f"\n  {i}. {step.action}")
        print(f"     Columns: {step.columns}")
        print(f"     Params:  {step.params}")
        print(f"     Reason:  {step.reason}")

    if plan.warnings:
        print(f"\n  Warnings:")
        for w in plan.warnings:
            print(f"     - {w}")

    if plan.notes:
        print(f"\n  Notes:")
        for n in plan.notes:
            print(f"     - {n}")

    # --------------------------------------------------
    # 2. Service: Deterministic execution of the plan
    # --------------------------------------------------

    print("\n========== FEATURE ENGINEERING SERVICE ==========\n")

    service = FeatureEngineeringService()
    transformed_df, report = service.apply_plan(df, plan)

    print(f"Original shape:   {report['original_shape']}")
    print(f"Final shape:      {report['final_shape']}")
    print(f"Strategy:         {report['strategy']}")
    print(f"Steps executed:   {report['steps_executed']}")
    print(f"Steps failed:     {report['steps_failed']}")

    print(f"\nColumns added:    {report['columns_added']}")
    print(f"Columns removed:  {report['columns_removed']}")

    print("\nStep Details:")
    for detail in report["step_details"]:
        status_icon = "✅" if detail["status"] == "success" else "❌"
        print(f"  {status_icon} {detail['action']}: {detail['details']}")
        if detail["status"] == "failed":
            print(f"     Error: {detail.get('error')}")

    # --------------------------------------------------
    # 3. Inspect result
    # --------------------------------------------------

    print("\n========== TRANSFORMED DATAFRAME ==========\n")
    print(transformed_df.head())
    print(f"\nShape: {transformed_df.shape}")
    print(f"\nDtypes:")
    print(transformed_df.dtypes)

    print("\n========== DONE ==========\n")


if __name__ == "__main__":
    asyncio.run(main())
