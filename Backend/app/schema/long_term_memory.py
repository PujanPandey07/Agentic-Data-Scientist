from datetime import datetime, timezone

from pydantic import BaseModel, Field


class LongTermMemory(BaseModel):
    """A selective durable fact, independent of one conversation's messages."""

    memory_id: str
    owner_id: str
    topic: str
    content: str
    importance: float = Field(ge=0.0, le=1.0)
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
