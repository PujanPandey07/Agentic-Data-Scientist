from pathlib import Path
from uuid import UUID, uuid4
import logging

from fastapi import HTTPException, UploadFile
import pandas as pd

from schema.upload import UploadResponse

logger = logging.getLogger(__name__)


class DatasetService:
    ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

    def __init__(self):
        base_dir = Path(__file__).resolve().parent.parent

        self.upload_dir = base_dir / "uploads"
        self.upload_dir.mkdir(exist_ok=True)

    async def upload_dataset(self, file: UploadFile) -> UploadResponse:
        extension = Path(file.filename).suffix.lower()

        if extension not in self.ALLOWED_EXTENSIONS:
            logger.warning(
                f"Rejected upload with unsupported extension: {extension}")
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type."
            )

        dataset_id = str(uuid4())
        stored_filename = f"{dataset_id}{extension}"

        file_path = self.upload_dir / stored_filename

        contents = await file.read()

        with open(file_path, "wb") as f:
            f.write(contents)

        logger.info(
            f"Uploaded dataset {dataset_id} ({file.filename}, {len(contents)} bytes)")

        return UploadResponse(
            success=True,
            dataset_id=dataset_id,
            filename=file.filename,
            stored_filename=stored_filename,
            file_size=len(contents),
            file_type=extension.replace(".", ""),
            mime_type=file.content_type,
        )

    def get_dataset_path(self, dataset_id: str) -> Path:
        try:
            canonical_id = str(UUID(dataset_id))
        except (AttributeError, ValueError):
            logger.error(f"Dataset not found: {dataset_id}")
            raise HTTPException(
                status_code=404,
                detail="Dataset not found."
            )

        for extension in self.ALLOWED_EXTENSIONS:
            file_path = self.upload_dir / f"{canonical_id}{extension}"
            if file_path.is_file():
                logger.info(
                    f"Resolved dataset {dataset_id} -> {file_path.name}")
                return file_path

        logger.error(f"Dataset not found: {dataset_id}")
        raise HTTPException(
            status_code=404,
            detail="Dataset not found."
        )

    def load_dataset(self, dataset_id: str) -> pd.DataFrame:

        file_path = self.get_dataset_path(dataset_id)

        extension = file_path.suffix.lower()

        if extension == ".csv":
            return pd.read_csv(file_path)

        elif extension in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)

        logger.error(f"Unsupported dataset type: {extension}")
        raise HTTPException(
            status_code=400,
            detail="Unsupported dataset type."
        )


dataset_service = DatasetService()
