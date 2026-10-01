import os
from pathlib import Path
from uuid import UUID, uuid4
import logging

from fastapi import HTTPException, UploadFile
import pandas as pd

from schema.upload import UploadResponse

logger = logging.getLogger(__name__)

# Configurable guardrails via environment variables
MAX_FILE_SIZE_MB = int(os.getenv("MAX_DATASET_SIZE_MB", "100"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_DATASET_ROWS = int(os.getenv("MAX_DATASET_ROWS", "500000"))
MAX_DATASET_COLUMNS = int(os.getenv("MAX_DATASET_COLUMNS", "1000"))
MIN_DATASET_COLUMNS = 2
MIN_DATASET_ROWS = 1


def read_file_to_dataframe(file_path: Path) -> pd.DataFrame:
    """
    Robust reader supporting major tabular file formats with graceful encoding fallbacks.
    """
    ext = file_path.suffix.lower()
    try:
        if ext == ".csv":
            try:
                return pd.read_csv(file_path)
            except (UnicodeDecodeError, pd.errors.ParserError):
                return pd.read_csv(file_path, encoding="latin1")

        elif ext in [".tsv", ".tab"]:
            try:
                return pd.read_csv(file_path, sep="\t")
            except (UnicodeDecodeError, pd.errors.ParserError):
                return pd.read_csv(file_path, sep="\t", encoding="latin1")

        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)

        elif ext in [".parquet", ".pq"]:
            return pd.read_parquet(file_path)

        elif ext == ".json":
            try:
                return pd.read_json(file_path)
            except ValueError:
                return pd.read_json(file_path, lines=True)

        elif ext == ".feather":
            return pd.read_feather(file_path)

        else:
            raise ValueError(f"Unsupported format '{ext}'")

    except Exception as e:
        logger.error(f"Error parsing {file_path.name}: {e}")
        raise ValueError(f"Could not parse {ext} file: {str(e)}") from e


class DatasetService:
    ALLOWED_EXTENSIONS = {
        ".csv",
        ".tsv",
        ".tab",
        ".xlsx",
        ".xls",
        ".parquet",
        ".pq",
        ".json",
        ".feather",
    }

    def __init__(self):
        base_dir = Path(__file__).resolve().parent.parent
        self.upload_dir = base_dir / "uploads"
        self.upload_dir.mkdir(exist_ok=True)

    async def upload_dataset(self, file: UploadFile) -> UploadResponse:
        filename = file.filename or "uploaded_dataset"
        extension = Path(filename).suffix.lower()

        if extension not in self.ALLOWED_EXTENSIONS:
            logger.warning(f"Rejected upload with unsupported extension: {extension}")
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file format '{extension}'. Supported formats: "
                    f"{', '.join(sorted(self.ALLOWED_EXTENSIONS))}"
                ),
            )

        dataset_id = str(uuid4())
        stored_filename = f"{dataset_id}{extension}"
        file_path = self.upload_dir / stored_filename

        total_bytes = 0
        chunk_size = 1024 * 1024  # 1MB chunks

        try:
            # Guardrail 1: Streamed file writing with strict size limit
            with open(file_path, "wb") as f:
                while chunk := await file.read(chunk_size):
                    total_bytes += len(chunk)
                    if total_bytes > MAX_FILE_SIZE_BYTES:
                        file_path.unlink(missing_ok=True)
                        logger.warning(
                            f"Rejected upload exceeding size limit: {total_bytes} bytes > {MAX_FILE_SIZE_BYTES} bytes"
                        )
                        raise HTTPException(
                            status_code=413,
                            detail=f"File exceeds the maximum allowed size of {MAX_FILE_SIZE_MB}MB.",
                        )
                    f.write(chunk)

            # Guardrail 2: Reject empty (0-byte) files
            if total_bytes == 0:
                file_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded file is empty (0 bytes).",
                )

            # Guardrail 3: Integrity & parse validation upon upload
            try:
                df = read_file_to_dataframe(file_path)
            except ValueError as err:
                file_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=400,
                    detail=f"Corrupt or invalid dataset: {str(err)}",
                )

            # Guardrail 4: Tabular structure & dimension checks
            if df.empty or len(df) < MIN_DATASET_ROWS:
                file_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=400,
                    detail="Dataset contains 0 rows of data.",
                )

            if len(df.columns) < MIN_DATASET_COLUMNS:
                file_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Dataset has only {len(df.columns)} column(s). "
                        f"A minimum of {MIN_DATASET_COLUMNS} columns is required to perform data analysis or ML modeling."
                    ),
                )

            if len(df.columns) > MAX_DATASET_COLUMNS:
                file_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Dataset has {len(df.columns)} columns, exceeding the maximum supported limit "
                        f"of {MAX_DATASET_COLUMNS} columns."
                    ),
                )

            if len(df) > MAX_DATASET_ROWS:
                file_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Dataset has {len(df):,} rows, exceeding the in-memory execution limit "
                        f"of {MAX_DATASET_ROWS:,} rows."
                    ),
                )

        except HTTPException:
            # Re-raise explicit HTTPExceptions after cleanup
            raise
        except Exception as e:
            file_path.unlink(missing_ok=True)
            logger.exception("Unexpected error during file upload processing")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process uploaded file: {str(e)}",
            )

        logger.info(
            f"Uploaded dataset {dataset_id} ({filename}, {total_bytes} bytes, {len(df)} rows, {len(df.columns)} cols)"
        )

        return UploadResponse(
            success=True,
            dataset_id=dataset_id,
            filename=filename,
            stored_filename=stored_filename,
            file_size=total_bytes,
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
                detail="Dataset not found.",
            )

        for extension in self.ALLOWED_EXTENSIONS:
            file_path = self.upload_dir / f"{canonical_id}{extension}"
            if file_path.is_file():
                logger.info(f"Resolved dataset {dataset_id} -> {file_path.name}")
                return file_path

        logger.error(f"Dataset not found: {dataset_id}")
        raise HTTPException(
            status_code=404,
            detail="Dataset not found.",
        )

    def load_dataset(self, dataset_id: str) -> pd.DataFrame:
        file_path = self.get_dataset_path(dataset_id)
        try:
            return read_file_to_dataframe(file_path)
        except ValueError as err:
            logger.error(f"Failed to load dataset {dataset_id}: {err}")
            raise HTTPException(
                status_code=400,
                detail=str(err),
            )


dataset_service = DatasetService()
