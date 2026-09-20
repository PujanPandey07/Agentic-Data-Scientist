from typing import TypedDict
import pandas as pd

from schema.model_selection import ModelSelectionPlan
from schema.feature_planner import FeatureEngineeringPlan
from schema.visualization_plan import VisualizationPlanResponse
from schema.analysis_plan import AnalysisPlan
from schema.dataset_summary import DatasetSummary


class GraphState(TypedDict):
    # User Input
    user_query: str
    base_user_query: str | None
    dataset_id: str
    conversation_id: str | None
    target_column: str | None
    intent: str | None
    direct_answer: str | None
    refine_target: str | None

    refine_instruction: str | None
    refine_confidence: float | None
    no_prior_analysis: bool | None
    refine_confirmed: bool | None
    plan_confirmed: bool | None
    plan_review_cancelled: bool | None
    fan_out_viz_fe: bool | None
    _viz_just_completed: str | None
    _fe_just_completed: str | None
    user_constraints: dict[str, list[str]] | None

    # Dataset
    # Large runtime artifacts (dataframe / train_df / test_df) are intentionally
    # kept out of the checkpointed graph state and stored in a memory cache keyed
    # by dataset_id so the persisted state stays lightweight.
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
    train_df: pd.DataFrame | None
    test_df: pd.DataFrame | None

    # Training
    model_selection_plan: ModelSelectionPlan | None
    training_report: dict | None
    trained_model_path: str | None   # NEW — set by training node
    _training_job_id: str | None

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
