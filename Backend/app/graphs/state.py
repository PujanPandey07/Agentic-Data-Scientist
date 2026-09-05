from typing import TypedDict
import pandas as pd

from schema.model_selection import ModelSelectionPlan
from schema.feature_planner import FeatureEngineeringPlan
from schema.feature_planner import FeatureEngineeringPlan
from schema.visualization_plan import VisualizationPlanResponse
from schema.analysis_plan import AnalysisPlan
from schema.dataset_summary import DatasetSummary


class GraphState(TypedDict):
    # User Input
    user_query: str
    dataset_id: str
    target_column: str | None
    intent: str | None
    direct_answer: str | None

    # Dataset
    dataframe: pd.DataFrame | None
    dataset_summary: DatasetSummary | None

    # Planner
    analysis_plan: AnalysisPlan | None

    # Cleaning
    cleaning_report: dict | None

    # EDA
    eda_report: dict | None

    # Visualization
    visualization_plan: VisualizationPlanResponse | None
    visualization_results: list[dict] | None

    # Feature Engineering
    feature_engineering_plan: FeatureEngineeringPlan | None
    feature_engineering_report: dict | None

    # Training
    model_selection_plan: ModelSelectionPlan | None
    training_report: dict | None
    trained_model_path: str | None   # NEW — set by training node

    # evaluation
    evaluation_report: dict | None
    hyperparameter_tuning_report: dict | None
    tuned_model_path: str | None

    # reporting
    final_report: dict | None

    # Task Management
    current_task: str | None
    remaining_tasks: list[str]
    completed_tasks: list[str]
    execution_logs: list[dict]
