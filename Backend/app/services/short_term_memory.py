import hashlib
import logging
from pathlib import Path

from cache.json_store import JsonCacheStore
from schema.conversation_memory import ConversationMessage, ShortTermMemory

logger = logging.getLogger(__name__)


class ShortTermMemoryManager:
    """Stores bounded conversational context without storing graph state."""

    MAX_MESSAGES = 12
    MAX_MESSAGE_CHARS = 4_000
    MAX_CONTEXT_CHARS = 16_000
    SUMMARY_MESSAGE_CHARS = 500
    MAX_DECISIONS = 10
    MAX_UNRESOLVED_QUESTIONS = 10

    def __init__(self, directory: Path | None = None):
        directory = directory or Path(__file__).resolve(
        ).parent.parent / "cache" / "data" / "conversations"
        self.store = JsonCacheStore(directory)

    def get(self, conversation_id: str) -> ShortTermMemory:
        value = self.store.get(self._key(conversation_id))
        if value is None:
            return ShortTermMemory(conversation_id=conversation_id)
        return ShortTermMemory.model_validate(value)

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> ShortTermMemory:
        memory = self.get(conversation_id)
        message = ConversationMessage(
            role=role,
            content=content[:self.MAX_MESSAGE_CHARS],
        )
        memory.recent_messages.append(message)
        self._manage_context(memory)
        self.set(memory)
        return memory

    def _manage_context(self, memory: ShortTermMemory) -> None:
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

        memory.recent_messages = memory.recent_messages[-self.MAX_MESSAGES:]

    def _message_chars(self, memory: ShortTermMemory) -> int:
        return sum(len(message.content) for message in memory.recent_messages)

    def _summarize(
        self,
        existing_summary: str,
        messages: list[ConversationMessage],
    ) -> str:
        new_points = " ".join(
            f"{message.role}: {message.content[:self.SUMMARY_MESSAGE_CHARS]}"
            for message in messages
        )
        combined = " ".join(part for part in (
            existing_summary, new_points) if part)
        return combined[-self.MAX_CONTEXT_CHARS:]

    def set(self, memory: ShortTermMemory) -> None:
        self.store.set(
            self._key(memory.conversation_id),
            memory.model_dump(mode="json"),
        )

    def update_facts(
        self,
        conversation_id: str,
        *,
        current_goal: str | None = None,
        decision: str | None = None,
        unresolved_question: str | None = None,
    ) -> ShortTermMemory:
        memory = self.get(conversation_id)
        if current_goal:
            memory.current_goal = current_goal
        if decision and decision not in memory.recent_decisions:
            memory.recent_decisions = (
                memory.recent_decisions + [decision]
            )[-self.MAX_DECISIONS:]
        if unresolved_question and unresolved_question not in memory.unresolved_questions:
            memory.unresolved_questions = (
                memory.unresolved_questions + [unresolved_question]
            )[-self.MAX_UNRESOLVED_QUESTIONS:]
        self.set(memory)
        return memory

    def invalidate(self, conversation_id: str) -> None:
        self.store.invalidate(self._key(conversation_id))

    @staticmethod
    def _key(conversation_id: str) -> str:
        digest = hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()
        return f"conversation_{digest}"


short_term_memory_manager = ShortTermMemoryManager()
