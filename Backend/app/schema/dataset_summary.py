from pydantic import BaseModel


class DatasetSummary(BaseModel):
    rows: int
    columns: int

    column_names: list[str]

    data_types: dict[str, str]

    numerical_columns: list[str]
    categorical_columns: list[str]

    missing_values: dict[str, int]

    duplicate_rows: int

    memory_usage: str

    has_missing_values: bool

    has_duplicates: bool

    potential_target_columns: list[str]
