from typing import TypedDict
import pandas as pd

from schema.analysis_plan import AnalysisPlan
from schema.dataset_summary import DatasetSummary


class GraphState(TypedDict):
    # User Input
    user_query: str
    dataset_id: str

    # Dataset
    dataframe: pd.DataFrame | None
    dataset_summary: DatasetSummary | None

    # Planner
    analysis_plan: AnalysisPlan | None

    # Cleaning
    cleaning_report: dict | None

    # EDA
    eda_report: dict | None

    # Task Management
    current_task: str | None
    remaining_tasks: list[str]
    completed_tasks: list[str]
    execution_logs: list[dict]
