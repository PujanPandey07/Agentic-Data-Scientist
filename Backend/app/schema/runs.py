# schema/runs.py
from typing import Any
from pydantic import BaseModel


class RunRequest(BaseModel):
    dataset_id: str
    user_query: str


class RunResponse(BaseModel):
    thread_id: str
    interrupted: bool
    # Raw interrupt payload if the graph paused (e.g. plan_review) —
    # we're not handling resume yet, just surfacing it as-is for now.
    interrupt: dict[str, Any] | None = None
    intent: str | None = None
    direct_answer: str | None = None
