# schema/chat.py
from typing import Any
from pydantic import BaseModel


class ResumeDecision(BaseModel):
    approved: bool
    edit_instruction: str | None = None


class ChatRequest(BaseModel):
    thread_id: str
    user_query: str | None = None       # used when NOT paused
    decision: ResumeDecision | None = None  # used when paused


class ChatResponse(BaseModel):
    interrupted: bool
    interrupt: dict[str, Any] | None = None
    intent: str | None = None
    direct_answer: str | None = None
