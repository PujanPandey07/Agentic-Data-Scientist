from fastapi import FastAPI

from api.health import router as health_router
from api.upload import router as upload_router
from api.analysis import router as analysis_router

app = FastAPI(
    title="AI Data Scientist",
    version="0.1.0",
    description="An Agentic AI Data Scientist built with LangGraph.",
)

app.include_router(health_router)
app.include_router(upload_router)
app.include_router(analysis_router)
