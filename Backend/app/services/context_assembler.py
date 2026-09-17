from cache.analysis_cache import analysis_context_cache
from schema.conversation_memory import ShortTermMemory
from services.long_term_memory import long_term_memory_manager
from services.short_term_memory import short_term_memory_manager


class LLMContextAssembler:
    """Builds intentional LLM context without exposing GraphState."""

    MAX_CONVERSATION_CHARS = 8_000
    MAX_ANALYSIS_CHARS = 12_000
    MAX_STRING_CHARS = 1_200
    MAX_LIST_ITEMS = 20
    MAX_DEPTH = 5

    async def assemble(
        self,
        conversation_id: str,
        query: str,
        dataset_id: str | None = None,
    ) -> dict:
        memory = await short_term_memory_manager.get(conversation_id)
        # NOT awaited — analysis_context_cache.get is a plain synchronous
        # in-memory cache lookup, unlike the two memory managers above.
        analysis = (
            analysis_context_cache.get(dataset_id)
            if dataset_id else None
        )
        long_term = await long_term_memory_manager.relevant(conversation_id, query)

        return {
            "conversation": self._compact(
                self._conversation_context(memory),
                self.MAX_CONVERSATION_CHARS,
            ),
            "analysis": self._compact(
                analysis.model_dump(mode="json") if analysis else None,
                self.MAX_ANALYSIS_CHARS,
            ),
            "long_term": [
                self._compact(item.model_dump(mode="json"), 1_000)
                for item in long_term
            ],
        }

    @classmethod
    def _compact(cls, value, budget: int, depth: int = 0):
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if depth >= cls.MAX_DEPTH:
            return str(value)[:cls.MAX_STRING_CHARS]
        if isinstance(value, str):
            return value[:min(cls.MAX_STRING_CHARS, budget)]
        if isinstance(value, list):
            result = []
            used = 2
            for item in value[:cls.MAX_LIST_ITEMS]:
                compact_item = cls._compact(
                    item, max(1, budget - used), depth + 1)
                item_size = len(str(compact_item)) + 1
                if used + item_size > budget and result:
                    break
                result.append(compact_item)
                used += item_size
            return result
        if isinstance(value, dict):
            result = {}
            used = 2
            for key, item in value.items():
                compact_item = cls._compact(
                    item, max(1, budget - used), depth + 1)
                item_size = len(str(key)) + len(str(compact_item)) + 4
                if used + item_size > budget and result:
                    break
                result[key] = compact_item
                used += item_size
            return result
        return str(value)[:min(cls.MAX_STRING_CHARS, budget)]

    @staticmethod
    def _conversation_context(memory: ShortTermMemory) -> dict:
        return {
            "summary": memory.conversation_summary,
            "recent_messages": [
                message.model_dump(mode="json")
                for message in memory.recent_messages[:-1]
            ],
            "current_goal": memory.current_goal,
            "recent_decisions": memory.recent_decisions,
            "unresolved_questions": memory.unresolved_questions,
        }


llm_context_assembler = LLMContextAssembler()
