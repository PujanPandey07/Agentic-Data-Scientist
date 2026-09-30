# core/models.py
from datetime import datetime, timezone
from sqlalchemy import ForeignKey, String, Text, Float, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from core.db import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(unique=True, index=True)
    hashed_password: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc))


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    thread_id: Mapped[str] = mapped_column(unique=True, index=True)
    dataset_id: Mapped[str]
    title: Mapped[str | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc))
    pipeline_family: Mapped[str] = mapped_column(String, default="supervised")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id"))
    role: Mapped[str]
    content: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc))


# added to core/models.py
class UserAPIKey(Base):
    __tablename__ = "user_api_keys"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), unique=True, index=True)
    provider: Mapped[str]          # "openai" | "anthropic" | "gemini"
    encrypted_key: Mapped[str]
    model_name: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc))


class LongTermMemoryRow(Base):
    """One durable memory fact per user per topic."""
    __tablename__ = "long_term_memories"

    # Deterministic SHA-256 of owner_id:topic — same as the old json key.
    memory_id: Mapped[str] = mapped_column(String, primary_key=True)
    owner_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    topic: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    importance: Mapped[float] = mapped_column(Float, nullable=False, default=0.7)
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        # Each owner can only have one entry per topic.
        UniqueConstraint("owner_id", "topic", name="uq_ltm_owner_topic"),
    )


class ShortTermMemoryRow(Base):
    """Bounded conversational context for one conversation thread."""
    __tablename__ = "short_term_memories"

    # conversation_id is the thread_id string from LangGraph.
    conversation_id: Mapped[str] = mapped_column(String, primary_key=True)
    conversation_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # JSONB stores list[{role, content}] — fast and queryable if ever needed.
    recent_messages: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    current_goal: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    recent_decisions: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    unresolved_questions: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

