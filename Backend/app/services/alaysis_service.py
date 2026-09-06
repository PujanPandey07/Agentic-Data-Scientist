import logging
from analysis.inspector import dataset_inspector
from services.dataset_service import dataset_service

logger = logging.getLogger(__name__)


class AnalysisService:

    async def inspect_dataset(self, dataset_id: str):
        logger.info(f"Inspecting dataset: {dataset_id}")

        dataframe = dataset_service.load_dataset(dataset_id)

        summary = dataset_inspector.inspect(dataframe)
        logger.info(
            f"Inspection done for {dataset_id}: {len(summary)} summary fields")

        return summary


analysis_service = AnalysisService()
