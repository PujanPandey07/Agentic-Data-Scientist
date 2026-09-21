import json
import redis.asyncio as aioredis
from arq import create_pool
from arq.connections import RedisSettings
from arq.jobs import Job, JobStatus

REDIS_HOST = "localhost"
REDIS_PORT = 6380

_pool = None
_redis = None  # raw redis client used only for pub/sub


async def get_pool():
    """Lazily create one shared arq Redis connection pool, reused across
    calls instead of opening a new connection every time a node runs."""
    global _pool
    if _pool is None:
        _pool = await create_pool(RedisSettings(host=REDIS_HOST, port=REDIS_PORT))
    return _pool


async def _get_redis():
    """Raw redis.asyncio client — used for pub/sub which arq's pool
    doesn't expose. Separate from the arq pool on purpose."""
    global _redis
    if _redis is None:
        _redis = aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    return _redis


async def enqueue_training_job(dataset_id: str, target_column: str, plan_dict: dict) -> str:
    pool = await get_pool()
    job = await pool.enqueue_job("train_model_job", dataset_id, target_column, plan_dict)
    return job.job_id


async def publish_job_complete(job_id: str, result: dict) -> None:
    """Called by the arq worker the moment training finishes. Stores the
    result directly (so callers don't depend on arq's own job.status()
    bookkeeping, which finalizes slightly AFTER this function runs — a
    real race with the SSE-triggered resume) and publishes on the job's
    pub/sub channel so the SSE endpoint can push the event immediately."""
    redis = await _get_redis()
    await redis.set(f"job_result:{job_id}", json.dumps(result), ex=3600)
    await redis.publish(
        f"job:{job_id}",
        json.dumps({"status": "complete", "job_id": job_id}),
    )


async def check_job_result(job_id: str):
    """Returns {'status': 'complete', 'result': ...} or
    {'status': 'failed', 'error': ...} once the job is done, or None
    if still running."""
    redis = await _get_redis()
    raw = await redis.get(f"job_result:{job_id}")
    if raw is not None:
        return {"status": "complete", "result": json.loads(raw)}

    # Only reachable if the job failed before ever reaching
    # publish_job_complete — check arq's own status for that case.
    pool = await get_pool()
    job = Job(job_id, pool)
    status = await job.status()
    if status == JobStatus.failed:
        info = await job.result_info()
        return {"status": "failed", "error": str(info.result) if info else "Unknown error"}

    return None


async def enqueue_tuning_job(dataset_id: str, target_column: str,
                             best_candidate_dict: dict, problem_type: str,
                             max_trials: int, time_budget_seconds: int) -> str:
    pool = await get_pool()
    job = await pool.enqueue_job(
        "tune_model_job", dataset_id, target_column, best_candidate_dict,
        problem_type, max_trials, time_budget_seconds,
    )
    return job.job_id
