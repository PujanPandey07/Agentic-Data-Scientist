import logging
from analysis.inspector import dataset_inspector
from services.dataset_service import dataset_service
from cache.dataset_cache import dataset_summary_cache

logger = logging.getLogger(__name__)


class AnalysisService:

    async def inspect_dataset(self, dataset_id: str):
        logger.info(f"Inspecting dataset: {dataset_id}")

        cached_summary = dataset_summary_cache.get(dataset_id)
        if cached_summary is not None:
            return cached_summary

        dataframe = dataset_service.load_dataset(dataset_id)

        summary = dataset_inspector.inspect(dataframe)
        dataset_summary_cache.set(dataset_id, summary)
        logger.info(
            f"Inspection done for {dataset_id}: "
            f"{summary.rows} rows, {summary.columns} columns")

        return summary


analysis_service = AnalysisService()
