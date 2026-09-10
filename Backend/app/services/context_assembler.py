from cache.analysis_cache import analysis_context_cache
from schema.conversation_memory import ShortTermMemory
from services.long_term_memory import long_term_memory_manager
from services.short_term_memory import short_term_memory_manager


class LLMContextAssembler:
    """Builds intentional LLM context without exposing GraphState."""

    def assemble(
        self,
        conversation_id: str,
        query: str,
        dataset_id: str | None = None,
    ) -> dict:
        memory = short_term_memory_manager.get(conversation_id)
        analysis = (
            analysis_context_cache.get(dataset_id)
            if dataset_id else None
        )
        long_term = long_term_memory_manager.relevant(conversation_id, query)

        return {
            "conversation": self._conversation_context(memory),
            "analysis": analysis.model_dump(mode="json") if analysis else None,
            "long_term": [item.model_dump(mode="json") for item in long_term],
        }

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
