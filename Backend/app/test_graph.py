import asyncio

from graphs.workflow import graph
from services.dataset_service import dataset_service


async def main():

    dataset_id = "08f054c7-0d54-4db4-95db-b84616f7bf25"

    state = {
        "dataset_id": dataset_id,
        "user_query": "Predict whether a customer will purchase a product.",
    }

    result = await graph.ainvoke(state)

    print("\n========== CLEANING RESULT ==========\n")

    print("Original/Final dataframe:")
    print(result["dataframe"])

    print("\n========== CLEANING REPORT ==========\n")
    print(result.get("cleaning_report"))

    print("\n========== COMPLETED TASKS ==========\n")
    print(result.get("completed_tasks"))

    print("\n========== CURRENT TASK ==========\n")
    print(result.get("current_task"))

    print("\n========== EXECUTION LOGS ==========\n")
    for log in result.get("execution_logs", []):
        print(log)


if __name__ == "__main__":
    asyncio.run(main())
