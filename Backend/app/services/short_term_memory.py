# services/short_term_memory.py
import asyncio
import logging
from collections import defaultdict

from cache.pg_store import PgShortTermStore
from schema.conversation_memory import ConversationMessage, ShortTermMemory

logger = logging.getLogger(__name__)


class ShortTermMemoryManager:
    """Stores bounded conversational context in Postgres.

    The public API is identical to the old JSON-backed version — callers
    need no changes. Only the storage backend changed from disk JSON to a
    Postgres row keyed by conversation_id.
    """

    MAX_MESSAGES = 12
    MAX_MESSAGE_CHARS = 4_000
    MAX_CONTEXT_CHARS = 16_000
    SUMMARY_MESSAGE_CHARS = 500
    MAX_DECISIONS = 10
    MAX_UNRESOLVED_QUESTIONS = 10

    def __init__(self) -> None:
        self.store = PgShortTermStore()
        # Serialise concurrent writes per conversation, allow parallel
        # writes across different conversations.
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def get(self, conversation_id: str) -> ShortTermMemory:
        data = await self.store.get(conversation_id)
        if data is None:
            return ShortTermMemory(conversation_id=conversation_id)
        return ShortTermMemory.model_validate(data)

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> ShortTermMemory:
        async with self._locks[conversation_id]:
            memory = await self.get(conversation_id)
            message = ConversationMessage(
                role=role,
                content=content[: self.MAX_MESSAGE_CHARS],
            )
            memory.recent_messages.append(message)
            self._manage_context(memory)
            await self.set(memory)
            return memory

    def _manage_context(self, memory: ShortTermMemory) -> None:
        """Trim oldest messages and fold them into the rolling summary."""
        overflow = max(0, len(memory.recent_messages) - self.MAX_MESSAGES)
        while overflow < len(memory.recent_messages) and self._message_chars(
            ShortTermMemory(
                conversation_id=memory.conversation_id,
                recent_messages=memory.recent_messages[overflow:],
            )
        ) > self.MAX_CONTEXT_CHARS:
            overflow += 1

        if overflow:
            older = memory.recent_messages[:overflow]
            memory.recent_messages = memory.recent_messages[overflow:]
            memory.conversation_summary = self._summarize(
                memory.conversation_summary, older
            )

        memory.recent_messages = memory.recent_messages[-self.MAX_MESSAGES :]

    def _message_chars(self, memory: ShortTermMemory) -> int:
        return sum(len(msg.content) for msg in memory.recent_messages)

    def _summarize(
        self,
        existing_summary: str,
        messages: list[ConversationMessage],
    ) -> str:
        new_points = " ".join(
            f"{msg.role}: {msg.content[:self.SUMMARY_MESSAGE_CHARS]}"
            for msg in messages
        )
        combined = " ".join(
            part for part in (existing_summary, new_points) if part
        )
        return combined[-self.MAX_CONTEXT_CHARS :]

    async def set(self, memory: ShortTermMemory) -> None:
        await self.store.set(memory.model_dump(mode="json"))

    async def update_facts(
        self,
        conversation_id: str,
        *,
        current_goal: str | None = None,
        decision: str | None = None,
        unresolved_question: str | None = None,
    ) -> ShortTermMemory:
        async with self._locks[conversation_id]:
            memory = await self.get(conversation_id)
            if current_goal:
                memory.current_goal = current_goal
            if decision and decision not in memory.recent_decisions:
                memory.recent_decisions = (
                    memory.recent_decisions + [decision]
                )[-self.MAX_DECISIONS :]
            if (
                unresolved_question
                and unresolved_question not in memory.unresolved_questions
            ):
                memory.unresolved_questions = (
                    memory.unresolved_questions + [unresolved_question]
                )[-self.MAX_UNRESOLVED_QUESTIONS :]
            await self.set(memory)
            return memory

    async def invalidate(self, conversation_id: str) -> None:
        await self.store.invalidate(conversation_id)


short_term_memory_manager = ShortTermMemoryManager()
