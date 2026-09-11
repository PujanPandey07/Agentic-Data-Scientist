# models.py
from datetime import datetime
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from db import Base


class Conversation(Base):
    __tablename__ = "conversations"

    # Django equivalent: id = models.AutoField(primary_key=True) — implicit in Django, explicit here
    id: Mapped[int] = mapped_column(primary_key=True)

    # thread_id is what LINKS this row to a checkpoint in the
    # LangGraph SQLite store — but there's no real foreign key here,
    # since that's a separate database entirely. It's just a string
    # we use to separately query the checkpointer when needed.
    thread_id: Mapped[str] = mapped_column(unique=True, index=True)

    dataset_id: Mapped[str]
    title: Mapped[str | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)

    # This IS a real foreign key — both tables live in the same
    # SQLite file, so a proper relational link works here.
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id"))

    role: Mapped[str]        # "user" or "assistant"
    content: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
