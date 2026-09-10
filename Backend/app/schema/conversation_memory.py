from typing import Literal

from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ShortTermMemory(BaseModel):
    """Bounded context for one conversation, separate from analysis state."""

    conversation_id: str
    conversation_summary: str = ""
    recent_messages: list[ConversationMessage] = Field(default_factory=list)
    current_goal: str | None = None
    recent_decisions: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
