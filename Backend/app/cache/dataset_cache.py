from pathlib import Path

from schema.dataset_summary import DatasetSummary

from cache.json_store import JsonCacheStore


class DatasetSummaryCache:
    """Cache for stable, dataset-level metadata only."""

    def __init__(self, directory: Path | None = None):
        directory = directory or Path(
            __file__).resolve().parent / "data" / "datasets"
        self.store = JsonCacheStore(directory)

    def get(self, dataset_id: str) -> DatasetSummary | None:
        value = self.store.get(self._key(dataset_id))
        return DatasetSummary.model_validate(value) if value else None

    def set(self, dataset_id: str, summary: DatasetSummary) -> None:
        self.store.set(self._key(dataset_id), summary.model_dump(mode="json"))

    def invalidate(self, dataset_id: str) -> None:
        self.store.invalidate(self._key(dataset_id))

    @staticmethod
    def _key(dataset_id: str) -> str:
        return f"dataset_summary_{dataset_id}"


dataset_summary_cache = DatasetSummaryCache()
