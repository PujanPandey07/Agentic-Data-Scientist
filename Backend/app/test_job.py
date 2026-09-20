import asyncio
from arq import create_pool
from arq.connections import RedisSettings

DATASET_ID = "191ce2e2-8419-4392-8e11-2e417a226482"
TARGET_COLUMN = "target"

plan_dict = {
    "strategy": "quick",
    "sample_size": None,
    "candidates": [
        {
            "algorithm": "logistic_regression",
            "reason": "Test run via background job",
            "estimated_time_seconds": 10,
            "hyperparams": {},
            "priority": 1,
        }
    ],
    "cv_folds": 3,
    "scoring_metric": "accuracy",
    "time_budget_minutes": 5,
    "notes": ["Manual test of train_model_job"],
    "forced_algorithm": None,
}


async def main():
    redis = await create_pool(RedisSettings(host="localhost", port=6380))
    job = await redis.enqueue_job(
        "train_model_job", DATASET_ID, TARGET_COLUMN, plan_dict
    )
    print("Job enqueued:", job.job_id)

    result = await job.result(timeout=60)
    print("Job result:", result)


asyncio.run(main())
