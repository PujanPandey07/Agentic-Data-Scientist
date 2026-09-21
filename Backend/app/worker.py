# worker.py — arq worker definition. Run separately from the main API
# process: `arq worker.WorkerSettings` (a different terminal/process
# entirely from `uvicorn main:app`).
from schema.model_selection import ModelCandidate
from services.hyper_parameters_tuning import HyperparameterTuningService
from services.runtime_state import runtime_state_store
from schema.model_selection import ModelSelectionPlan
from services.trainning import TrainingService
from services.job_queue import publish_job_complete
import os
from arq.connections import RedisSettings
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = "localhost"
REDIS_PORT = 6380


async def startup(ctx):
    print("arq worker starting up")


async def shutdown(ctx):
    print("arq worker shutting down")


# Placeholder job — just to prove the worker can receive and run a real
# job end-to-end before we wire up actual training. ctx is arq's context
# dict (holds the redis connection, startup state, etc.) — every job
# function takes it as the first argument, even if unused.
async def ping(ctx):
    print("ping job executed")
    return "pong"


async def train_model_job(ctx, dataset_id: str, target_column: str, plan_dict: dict):
    """Runs TrainingService.train() in the background worker, using the
    same runtime_state_store (Redis + Parquet) the main app already uses
    — so the worker sees the exact same cached train split the API
    process wrote, even though it's a completely separate process.
    """
    train_df, _ = runtime_state_store.get_train_test(dataset_id)
    if train_df is None:
        raise ValueError(
            f"No cached train split found for dataset_id={dataset_id}")

    plan = ModelSelectionPlan(**plan_dict)

    service = TrainingService()
    final_model, report, best_candidate = service.train(
        df=train_df, plan=plan, target_column=target_column, dataset_id=dataset_id,
    )

    # Training is done — shout it on the job's Redis pub/sub channel.
    # The SSE endpoint (/api/jobs/{job_id}/stream) is subscribed and
    # waiting; this one publish call is all it takes to wake up the
    # frontend instantly, with zero polling.
    job_id = ctx["job_id"]
    await publish_job_complete(job_id, report)

    # The model is already saved to disk by TrainingService (model_path is
    # in `report`). We don't need to return the model object itself — just
    # the report, which is everything training_node needs to update state.
    return report


async def tune_model_job(ctx, dataset_id: str, target_column: str,
                         best_candidate_dict: dict, problem_type: str,
                         max_trials: int, time_budget_seconds: int):
    """Runs HyperparameterTuningService.tune() in the background worker,
    same pattern as train_model_job — fetches the cached train split via
    runtime_state_store rather than shipping the DataFrame through Redis."""
    train_df, _ = runtime_state_store.get_train_test(dataset_id)
    if train_df is None:
        raise ValueError(
            f"No cached train split found for dataset_id={dataset_id}")

    best_candidate = ModelCandidate(**best_candidate_dict)

    service = HyperparameterTuningService()
    final_model, report = service.tune(
        df=train_df,
        target_column=target_column,
        best_candidate=best_candidate,
        problem_type=problem_type,
        dataset_id=dataset_id,
        max_trials=max_trials,
        time_budget_seconds=time_budget_seconds,
    )

    job_id = ctx["job_id"]
    await publish_job_complete(job_id, report)

    return report


class WorkerSettings:
    functions = [ping, train_model_job, tune_model_job]
    redis_settings = RedisSettings(host=REDIS_HOST, port=REDIS_PORT)
    on_startup = startup
    on_shutdown = shutdown
