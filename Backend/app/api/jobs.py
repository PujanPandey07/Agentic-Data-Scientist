# api/jobs.py — SSE endpoint for background job status.
#
# Why a query-param token instead of the usual Authorization header?
# The browser's native EventSource API has no way to set custom headers
# (it's a platform limitation, not our choice). Passing the token in the
# URL is the standard workaround — it's still HTTPS-encrypted in transit,
# just visible in server logs, which is acceptable for short-lived tokens.
import json

import redis.asyncio as aioredis
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from core.security import decode_token
from services.job_queue import check_job_result, REDIS_HOST, REDIS_PORT

# How long (seconds) we'll keep the SSE connection open waiting for a job.
# Training rarely exceeds this; if it does, the frontend gets a "timeout"
# event and can show the user a helpful message.
JOB_STREAM_TIMEOUT = 600  # 10 minutes

router = APIRouter(prefix="/api", tags=["Jobs"])


@router.get("/jobs/{job_id}/stream")
async def stream_job_status(
    job_id: str,
    token: str = Query(...,
                       description="Access token (Bearer value without 'Bearer ' prefix)"),
):
    """SSE endpoint — the frontend connects here right after receiving a
    job_status interrupt. We subscribe to the Redis pub/sub channel for
    this job and push one event the instant the arq worker finishes.

    Flow:
        1. Validate the access token (same JWT the rest of the API uses).
        2. Check if the job already finished before we even subscribed
           (handles the race where training completes very quickly).
        3. If not done yet, subscribe to Redis channel `job:{job_id}` and
           yield a keepalive comment every 15 s to prevent proxy timeouts.
        4. When the "complete" message arrives, send the SSE event and close.
        5. If JOB_STREAM_TIMEOUT expires, send a "timeout" event so the
           frontend can show a warning instead of hanging silently.
    """
    # Validate token — decode_token raises HTTPException(401) on failure.
    decode_token(token, expected_type="access")

    async def event_generator():
        redis = aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        pubsub = redis.pubsub()
        await pubsub.subscribe(f"job:{job_id}")

        try:
            # Subscribed FIRST — now check if it was already done. Even if
            # the job finishes in the instant between subscribing and this
            # check, the "complete" message will still arrive via the
            # subscription below, so nothing is missed either way.
            result = await check_job_result(job_id)
            if result is not None:
                yield f"data: {json.dumps({'status': 'complete', 'job_id': job_id})}\n\n"
                return

            elapsed = 0
            while elapsed < JOB_STREAM_TIMEOUT:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=15
                )

                if message is not None and message["type"] == "message":
                    yield f"data: {message['data'].decode()}\n\n"
                    return

                yield ": keepalive\n\n"
                elapsed += 15

            yield f"data: {json.dumps({'status': 'timeout', 'job_id': job_id})}\n\n"

        finally:
            await pubsub.unsubscribe(f"job:{job_id}")
            await redis.aclose()
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            # Prevent any caching of the event stream.
            "Cache-Control": "no-cache",
            # Tell proxies not to buffer — events must reach the browser
            # in real-time, not in one big buffered chunk at the end.
            "X-Accel-Buffering": "no",
        },
    )
