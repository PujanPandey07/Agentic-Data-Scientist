# schema/conversations.py
from datetime import datetime
from pydantic import BaseModel


class ConversationSummary(BaseModel):
    thread_id: str
    dataset_id: str
    title: str | None
    created_at: datetime


class ConversationListResponse(BaseModel):
    conversations: list[ConversationSummary]


class MessageInfo(BaseModel):
    role: str
    content: str
    created_at: datetime


class MessageListResponse(BaseModel):
    thread_id: str
    messages: list[MessageInfo]
