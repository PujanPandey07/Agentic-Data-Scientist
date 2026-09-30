# services/long_term_memory.py
import asyncio
import hashlib
import logging
import re
from collections import defaultdict

from cache.pg_store import PgLongTermStore
from schema.long_term_memory import LongTermMemory

logger = logging.getLogger(__name__)


class LongTermMemoryManager:
    """Selective durable memory with topic-based replacement and deletion.

    Backed by Postgres (via PgLongTermStore) instead of the old JSON file cache.
    The public API is identical — callers need no changes.
    """

    def __init__(self) -> None:
        self.store = PgLongTermStore()

        # One asyncio.Lock per owner_id — prevents concurrent read-modify-write
        # races for the same user while allowing different users to run fully
        # in parallel.
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def list(self, owner_id: str) -> list[LongTermMemory]:
        rows = await self.store.get_all(owner_id)
        return [LongTermMemory.model_validate(row) for row in rows]

    async def upsert(
        self,
        owner_id: str,
        topic: str,
        content: str,
        importance: float = 0.7,
    ) -> LongTermMemory:
        # Serialise all writes for this owner to prevent races.
        async with self._locks[owner_id]:
            memory_id = self._memory_id(owner_id, topic)
            row = await self.store.upsert(
                memory_id=memory_id,
                owner_id=owner_id,
                topic=topic,
                content=content,
                importance=importance,
            )
            return LongTermMemory.model_validate(row)

    async def delete(self, owner_id: str, topic: str) -> None:
        async with self._locks[owner_id]:
            await self.store.delete_by_topic(owner_id, topic)

    async def relevant(
        self, owner_id: str, query: str, limit: int = 5
    ) -> list[LongTermMemory]:
        terms = set(query.lower().split())
        scored = []
        for memory in await self.list(owner_id):
            score = len(terms.intersection(set(memory.content.lower().split())))
            scored.append((score, memory.importance, memory))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored[:limit] if item[0] > 0 or not terms]

    async def extract_and_store(
        self, owner_id: str, text: str
    ) -> list[LongTermMemory]:
        """Store only explicit durable preferences or project decisions."""
        candidates = []
        patterns = {
            "preference": r"(?:i prefer|user prefers|always use)\s+(.+?)(?:[.!?]|$)",
            "decision": r"(?:use|choose)\s+(.+?)\s+(?:instead|for this project)(?:[.!?]|$)",
        }
        for topic, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                candidates.append(
                    await self.upsert(owner_id, topic, match.group(1).strip(), 0.8)
                )
        return candidates

    @staticmethod
    def _memory_id(owner_id: str, topic: str) -> str:
        return hashlib.sha256(f"{owner_id}:{topic}".encode("utf-8")).hexdigest()


long_term_memory_manager = LongTermMemoryManager()
