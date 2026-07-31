from analysis.inspector import dataset_inspector
from services.dataset_service import dataset_service


class AnalysisService:

    async def inspect_dataset(self, dataset_id: str):

        dataframe = dataset_service.load_dataset(dataset_id)

        summary = dataset_inspector.inspect(dataframe)

        return summary


analysis_service = AnalysisService()
