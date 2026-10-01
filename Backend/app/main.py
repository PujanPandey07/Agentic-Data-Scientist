import logging
import logging.config
import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from graphs.workflow import build_supervised_graph, build_unsupervised_graph, build_time_series_graph
import os
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from core.db import engine, async_session, Base
from api.health import router as health_router
from api.upload import router as upload_router
from api.analysis import router as analysis_router
from api.runs import router as runs_router
from api.chat import router as chat_router
from api.reports import router as reports_router
from api.auth import router as auth_router
from api.conversations import router as conversations_router
from api.jobs import router as jobs_router
from api.api_key import router as api_key_router
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Logging — stdout so `docker compose logs` works out of the box.
# Previously went to app.log (a file), which is invisible inside a container.
# ---------------------------------------------------------------------------
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        },
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "default",
        },
    },
    "root": {"level": "INFO", "handlers": ["stdout"]},
    # Silence noisy libraries
    "loggers": {
        "uvicorn.access": {"level": "WARNING"},
        "httpx": {"level": "WARNING"},
        "httpcore": {"level": "WARNING"},
    },
})

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Startup cache sweep — delete runtime_cache Parquet files older than 24h
# so the volume doesn't grow unbounded over time.
# ---------------------------------------------------------------------------
_CACHE_MAX_AGE_SECONDS = 24 * 60 * 60   # 24 hours


def _sweep_runtime_cache() -> None:
    """Delete Parquet files in runtime_cache that are older than 24 hours.
    Called once at startup so stale files from before a container restart
    are cleaned up immediately. The ARQ worker runs this on a schedule too.
    """
    from services.runtime_state import CACHE_DIR
    try:
        cutoff = time.time() - _CACHE_MAX_AGE_SECONDS
        removed = 0
        for f in Path(CACHE_DIR).glob("*.parquet"):
            if f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
                removed += 1
        if removed:
            logger.info("Startup cache sweep: removed %d stale Parquet file(s)", removed)
    except Exception:
        logger.warning("Startup cache sweep failed (non-fatal)", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Clean up stale cache files before accepting traffic
    _sweep_runtime_cache()

    async with AsyncPostgresSaver.from_conn_string(
        os.getenv("DATABASE_URL_PSYCOPG")
    ) as checkpointer:
        await checkpointer.setup()

        # Two compiled graphs sharing the SAME checkpointer — this is what
        # lets both families' checkpoints live in the same Postgres store,
        # keyed by thread_id as always. Which graph a given thread_id gets
        # dispatched to is decided in api/runs.py (at creation) and looked
        # up via Conversation.pipeline_family in api/chat.py (on every
        # later message) — never decided here.
        graph = build_supervised_graph().compile(checkpointer=checkpointer)
        unsupervised_graph = build_unsupervised_graph().compile(checkpointer=checkpointer)
        time_series_graph = build_time_series_graph().compile(checkpointer=checkpointer)

        async with engine.begin() as db_conn:
            await db_conn.run_sync(Base.metadata.create_all)

        # Run Alembic migrations automatically on startup so new columns/tables exist
        from core.migrations import apply_migrations_async
        await apply_migrations_async()

        app.state.graph = graph
        app.state.unsupervised_graph = unsupervised_graph
        app.state.time_series_graph = time_series_graph
        app.state.db_session = async_session

        yield

    await engine.dispose()


app = FastAPI(
    title="AI Data Scientist",
    version="0.1.0",
    description="An Agentic AI Data Scientist built with LangGraph.",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Global exception handler — catches unhandled exceptions that slip past
# individual endpoint try/except blocks and returns a clean JSON 500
# instead of a raw Python traceback leaking to the client.
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "Unhandled exception on %s %s", request.method, request.url.path
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred. Please try again."},
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="outputs"), name="static")

app.include_router(health_router)
app.include_router(upload_router)
app.include_router(analysis_router)
app.include_router(runs_router)
app.include_router(chat_router)
app.include_router(reports_router)
app.include_router(auth_router)
app.include_router(conversations_router)
app.include_router(jobs_router)
app.include_router(api_key_router)
