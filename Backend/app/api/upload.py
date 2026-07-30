from fastapi import APIRouter, File, UploadFile

from schema.upload import UploadResponse
from services.dataset_service import dataset_service

router = APIRouter(
    prefix="/api",
    tags=["Dataset"],
)


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload a dataset",
)
async def upload_dataset(file: UploadFile = File(...)):
    return await dataset_service.upload_dataset(file)
