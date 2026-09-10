from pathlib import Path

from schema.analysis_context import AnalysisContext

from cache.json_store import JsonCacheStore


class AnalysisContextCache:
    """Cache for the latest completed analysis context per dataset."""

    def __init__(self, directory: Path | None = None):
        directory = directory or Path(
            __file__).resolve().parent / "data" / "analyses"
        self.store = JsonCacheStore(directory)

    def get(self, dataset_id: str) -> AnalysisContext | None:
        value = self.store.get(self._key(dataset_id))
        return AnalysisContext.model_validate(value) if value else None

    def set(self, dataset_id: str, context: AnalysisContext) -> None:
        self.store.set(self._key(dataset_id), context.model_dump(mode="json"))

    def invalidate(self, dataset_id: str) -> None:
        self.store.invalidate(self._key(dataset_id))

    @staticmethod
    def _key(dataset_id: str) -> str:
        return f"analysis_context_{dataset_id}"


analysis_context_cache = AnalysisContextCache()
