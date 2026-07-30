from pydantic import BaseModel


class UploadResponse(BaseModel):
    success: bool
    dataset_id: str
    filename: str
    stored_filename: str
    file_size: int
    file_type: str
    mime_type: str
