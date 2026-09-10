import hashlib
import logging
import re
from pathlib import Path

from cache.json_store import JsonCacheStore
from schema.long_term_memory import LongTermMemory

logger = logging.getLogger(__name__)


class LongTermMemoryManager:
    """Selective durable memory with topic-based replacement and deletion."""

    def __init__(self, directory: Path | None = None):
        directory = directory or Path(__file__).resolve(
        ).parent.parent / "cache" / "data" / "long_term"
        self.store = JsonCacheStore(directory)

    def list(self, owner_id: str) -> list[LongTermMemory]:
        value = self.store.get(self._key(owner_id))
        if not value:
            return []
        return [LongTermMemory.model_validate(item) for item in value.get("memories", [])]

    def upsert(
        self,
        owner_id: str,
        topic: str,
        content: str,
        importance: float = 0.7,
    ) -> LongTermMemory:
        memories = [memory for memory in self.list(
            owner_id) if memory.topic != topic]
        memory = LongTermMemory(
            memory_id=self._memory_id(owner_id, topic),
            owner_id=owner_id,
            topic=topic,
            content=content,
            importance=importance,
        )
        memories.append(memory)
        self.store.set(
            self._key(owner_id),
            {"memories": [item.model_dump(mode="json") for item in memories]},
        )
        return memory

    def delete(self, owner_id: str, topic: str) -> None:
        memories = [memory for memory in self.list(
            owner_id) if memory.topic != topic]
        self.store.set(
            self._key(owner_id),
            {"memories": [item.model_dump(mode="json") for item in memories]},
        )

    def relevant(self, owner_id: str, query: str, limit: int = 5) -> list[LongTermMemory]:
        terms = set(query.lower().split())
        scored = []
        for memory in self.list(owner_id):
            score = len(terms.intersection(
                set(memory.content.lower().split())))
            scored.append((score, memory.importance, memory))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored[:limit] if item[0] > 0 or not terms]

    def extract_and_store(self, owner_id: str, text: str) -> list[LongTermMemory]:
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
                    self.upsert(owner_id, topic, match.group(1).strip(), 0.8)
                )
        return candidates

    @staticmethod
    def _key(owner_id: str) -> str:
        return f"long_term_{hashlib.sha256(owner_id.encode('utf-8')).hexdigest()}"

    @staticmethod
    def _memory_id(owner_id: str, topic: str) -> str:
        return hashlib.sha256(f"{owner_id}:{topic}".encode("utf-8")).hexdigest()


long_term_memory_manager = LongTermMemoryManager()
