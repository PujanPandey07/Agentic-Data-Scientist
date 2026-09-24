from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from graphs.workflow import builder
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncPostgresSaver.from_conn_string(
        os.getenv("DATABASE_URL_PSYCOPG")
    ) as checkpointer:
        await checkpointer.setup()

        graph = builder.compile(checkpointer=checkpointer)

        async with engine.begin() as db_conn:
            await db_conn.run_sync(Base.metadata.create_all)

        app.state.graph = graph
        app.state.db_session = async_session

        yield

    await engine.dispose()


app = FastAPI(
    title="AI Data Scientist",
    version="0.1.0",
    description="An Agentic AI Data Scientist built with LangGraph.",
    lifespan=lifespan,
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

