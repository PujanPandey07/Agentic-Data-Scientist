# cache/pg_store.py
"""
Postgres-backed replacements for JsonCacheStore.

Two separate store classes — one per memory type — so each maps directly
onto its own table and uses native column types instead of a blob.

Both classes expose exactly the same async interface so the manager
classes (LongTermMemoryManager, ShortTermMemoryManager) need zero logic
changes — only their `self.store` assignment changes.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from core.db import async_session
from core.models import LongTermMemoryRow, ShortTermMemoryRow

logger = logging.getLogger(__name__)


# ─── Long-term memory store ──────────────────────────────────────────────────

class PgLongTermStore:
    """
    Drop-in async replacement for JsonCacheStore used by LongTermMemoryManager.

    The "key" passed by the manager is the owner_id string directly (not the
    hashed filename key — we store the real owner_id in the DB so we can
    index and query it efficiently).
    """

    async def get_all(self, owner_id: str) -> list[dict[str, Any]]:
        """Return all memory rows for owner as plain dicts (matches old JSON shape)."""
        async with async_session() as session:
            result = await session.execute(
                select(LongTermMemoryRow).where(LongTermMemoryRow.owner_id == owner_id)
            )
            rows = result.scalars().all()
            return [self._row_to_dict(row) for row in rows]

    async def upsert(self, memory_id: str, owner_id: str, topic: str,
                     content: str, importance: float) -> dict[str, Any]:
        """Insert or replace a single memory fact."""
        now = datetime.now(timezone.utc)
        stmt = (
            pg_insert(LongTermMemoryRow)
            .values(
                memory_id=memory_id,
                owner_id=owner_id,
                topic=topic,
                content=content,
                importance=importance,
                updated_at=now,
            )
            .on_conflict_do_update(
                index_elements=["memory_id"],
                set_={
                    "topic": topic,
                    "content": content,
                    "importance": importance,
                    "updated_at": now,
                },
            )
            .returning(LongTermMemoryRow)
        )
        async with async_session() as session:
            result = await session.execute(stmt)
            await session.commit()
            row = result.scalar_one()
            return self._row_to_dict(row)

    async def delete_by_topic(self, owner_id: str, topic: str) -> None:
        """Delete the memory for a specific owner+topic pair."""
        async with async_session() as session:
            await session.execute(
                delete(LongTermMemoryRow).where(
                    LongTermMemoryRow.owner_id == owner_id,
                    LongTermMemoryRow.topic == topic,
                )
            )
            await session.commit()

    @staticmethod
    def _row_to_dict(row: LongTermMemoryRow) -> dict[str, Any]:
        return {
            "memory_id": row.memory_id,
            "owner_id": row.owner_id,
            "topic": row.topic,
            "content": row.content,
            "importance": row.importance,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }


# ─── Short-term memory store ─────────────────────────────────────────────────

class PgShortTermStore:
    """
    Drop-in async replacement for JsonCacheStore used by ShortTermMemoryManager.

    The conversation_id is the primary key — one row per conversation thread.
    """

    async def get(self, conversation_id: str) -> dict[str, Any] | None:
        """Return the conversation memory row as a plain dict, or None if missing."""
        async with async_session() as session:
            result = await session.execute(
                select(ShortTermMemoryRow).where(
                    ShortTermMemoryRow.conversation_id == conversation_id
                )
            )
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return self._row_to_dict(row)

    async def set(self, data: dict[str, Any]) -> None:
        """Insert or fully replace the short-term memory for a conversation."""
        now = datetime.now(timezone.utc)
        stmt = (
            pg_insert(ShortTermMemoryRow)
            .values(
                conversation_id=data["conversation_id"],
                conversation_summary=data.get("conversation_summary", ""),
                recent_messages=data.get("recent_messages", []),
                current_goal=data.get("current_goal"),
                recent_decisions=data.get("recent_decisions", []),
                unresolved_questions=data.get("unresolved_questions", []),
                updated_at=now,
            )
            .on_conflict_do_update(
                index_elements=["conversation_id"],
                set_={
                    "conversation_summary": data.get("conversation_summary", ""),
                    "recent_messages": data.get("recent_messages", []),
                    "current_goal": data.get("current_goal"),
                    "recent_decisions": data.get("recent_decisions", []),
                    "unresolved_questions": data.get("unresolved_questions", []),
                    "updated_at": now,
                },
            )
        )
        async with async_session() as session:
            await session.execute(stmt)
            await session.commit()

    async def invalidate(self, conversation_id: str) -> None:
        """Delete the entire short-term memory row for a conversation."""
        async with async_session() as session:
            await session.execute(
                delete(ShortTermMemoryRow).where(
                    ShortTermMemoryRow.conversation_id == conversation_id
                )
            )
            await session.commit()

    @staticmethod
    def _row_to_dict(row: ShortTermMemoryRow) -> dict[str, Any]:
        return {
            "conversation_id": row.conversation_id,
            "conversation_summary": row.conversation_summary,
            "recent_messages": row.recent_messages or [],
            "current_goal": row.current_goal,
            "recent_decisions": row.recent_decisions or [],
            "unresolved_questions": row.unresolved_questions or [],
        }
