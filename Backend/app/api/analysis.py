from fastapi import APIRouter

from schema.dataset_summary import DatasetSummary
from services.alaysis_service import analysis_service

router = APIRouter(
    prefix="/api",
    tags=["Analysis"],
)


@router.get(
    "/analyze/{dataset_id}",
    response_model=DatasetSummary,
    summary="Inspect a dataset",
)
async def inspect_dataset(dataset_id: str):

    return await analysis_service.inspect_dataset(dataset_id)
