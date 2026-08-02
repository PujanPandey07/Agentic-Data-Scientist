import asyncio
from pprint import pprint

from graphs.workflow import graph


async def main():

    state = {
        "user_query": "Predict whether a customer will purchase a product.",
        # Replace this with a real dataset_id returned by your upload endpoint
        "dataset_id": "08f054c7-0d54-4db4-95db-b84616f7bf25",

        # These will be populated by the Dataset Node
        "dataframe": None,
        "dataset_summary": None,

        # This will be populated by the Planner Node
        "analysis_plan": None,
    }

    result = await graph.ainvoke(state)

    print("\n========== GRAPH STATE ==========\n")
    pprint(result)

    print("\n========== ANALYSIS PLAN ==========\n")
    print(result["analysis_plan"])


if __name__ == "__main__":
    asyncio.run(main())
