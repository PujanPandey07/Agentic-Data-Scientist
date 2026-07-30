from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile

from schema.upload import UploadResponse


class DatasetService:
    ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

    def __init__(self):
        self.upload_dir = Path("uploads")
        self.upload_dir.mkdir(exist_ok=True)

    async def upload_dataset(self, file: UploadFile) -> UploadResponse:
        extension = Path(file.filename).suffix.lower()

        if extension not in self.ALLOWED_EXTENSIONS:
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

        return UploadResponse(
            success=True,
            dataset_id=dataset_id,
            filename=file.filename,
            stored_filename=stored_filename,
            file_size=len(contents),
            file_type=extension.replace(".", ""),
            mime_type=file.content_type,
        )


dataset_service = DatasetService()
