# main.py — full updated file
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import aiosqlite

from graphs.workflow import builder
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from core.db import engine, async_session, Base
from api.health import router as health_router
from api.upload import router as upload_router
from api.analysis import router as analysis_router
from api.runs import router as runs_router
from api.chat import router as chat_router
from api.reports import router as reports_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    conn = await aiosqlite.connect("checkpoints.sqlite")
    checkpointer = AsyncSqliteSaver(
        conn, serde=JsonPlusSerializer(pickle_fallback=True))
    graph = builder.compile(checkpointer=checkpointer)

    async with engine.begin() as db_conn:
        await db_conn.run_sync(Base.metadata.create_all)

    app.state.graph = graph
    app.state.checkpoint_conn = conn
    app.state.db_session = async_session

    yield

    await conn.close()
    await engine.dispose()


app = FastAPI(
    title="AI Data Scientist",
    version="0.1.0",
    description="An Agentic AI Data Scientist built with LangGraph.",
    lifespan=lifespan,
)

# Exposes everything under outputs/ (charts, reports, evaluation plots,
# models) at URLs like http://host/static/charts/<dataset_id>/x.png.
# Frontend uses these directly in <img src="..."> tags or download links.
app.mount("/static", StaticFiles(directory="outputs"), name="static")

app.include_router(health_router)
app.include_router(upload_router)
app.include_router(analysis_router)
app.include_router(runs_router)
app.include_router(chat_router)
app.include_router(reports_router)
