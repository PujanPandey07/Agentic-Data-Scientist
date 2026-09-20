from arq import create_pool
from arq.connections import RedisSettings
from arq.jobs import Job

REDIS_HOST = "localhost"
REDIS_PORT = 6380

_pool = None


async def get_pool():
    """Lazily create one shared arq Redis connection pool, reused across
    calls instead of opening a new connection every time a node runs."""
    global _pool
    if _pool is None:
        _pool = await create_pool(RedisSettings(host=REDIS_HOST, port=REDIS_PORT))
    return _pool


async def enqueue_training_job(dataset_id: str, target_column: str, plan_dict: dict) -> str:
    pool = await get_pool()
    job = await pool.enqueue_job("train_model_job", dataset_id, target_column, plan_dict)
    return job.job_id


async def check_job_result(job_id: str):
    """Returns the job's result dict if finished, or None if still
    running/pending. Never blocks/waits — this is a one-shot check,
    called repeatedly from training_node on each resume."""
    pool = await get_pool()
    job = Job(job_id, pool)
    status = await job.status()
    if status == "complete":
        return await job.result()
    return None
