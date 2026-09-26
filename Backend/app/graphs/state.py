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
    llm_config: dict | None

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
    user_id: int | None

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

    # Training (supervised: classification/regression)
    model_selection_plan: ModelSelectionPlan | None
    training_report: dict | None
    trained_model_path: str | None   # NEW — set by training node
    _training_job_id: str | None

    # evaluation (supervised)
    evaluation_report: dict | None

    hyperparameter_tuning_report: dict | None
    tuned_model_path: str | None
    # Was NEVER declared here — enqueue_tuning_node/poll_tuning_node wrote
    # to it, but LangGraph only persists fields that exist on GraphState,
    # so every write silently vanished the instant enqueue_tuning_node
    # returned. poll_tuning_node then always read None, never called
    # interrupt(), and router kept redispatching straight back to
    # enqueue_tuning_node — an infinite loop (each iteration DID enqueue
    # a real background job) until LangGraph's recursion limit killed it.
    _tuning_job_id: str | None

    # --- Clustering (unsupervised) ---
    # Mirrors the supervised training/tuning/evaluation fields above in
    # shape (plan -> report -> path, enqueue/poll job id, separate
    # evaluation report), but kept as distinct keys rather than reusing
    # the supervised ones — a clustering run's "model_selection_plan"
    # holds a totally different schema (chosen algorithm + its specific
    # hyperparameter search space, e.g. n_clusters or eps/min_samples,
    # not classification/regression candidates), and reusing the same
    # key across both graph families would make it ambiguous which shape
    # a given checkpoint's value is in in every downstream node.
    clustering_model_selection_plan: dict | None
    clustering_training_report: dict | None
    clustering_model_path: str | None
    _clustering_job_id: str | None

    # Tuning objective/results are clustering-quality metrics (silhouette,
    # Davies-Bouldin, etc.) over a hyperparameter sweep (e.g. n_clusters),
    # not CV accuracy — kept separate from hyperparameter_tuning_report
    # for the same reason as above.
    clustering_tuning_report: dict | None
    tuned_clustering_model_path: str | None
    _clustering_tuning_job_id: str | None

    # Final cluster assignments for the dataset actually used to fit —
    # stored here (a plain list) rather than only inside a report dict,
    # since visualization (cluster scatter) and reporting both need direct
    # access to per-row labels, not just aggregate metrics.
    cluster_labels: list[int] | None

    # Clustering-quality metrics report: silhouette score, Davies-Bouldin
    # index, Calinski-Harabasz index, cluster size distribution, and (for
    # DBSCAN) noise-point count. Structurally unlike evaluation_report,
    # which assumes accuracy/F1/RMSE/R² — kept as its own key rather than
    # overloading evaluation_report with an incompatible shape.
    clustering_evaluation_report: dict | None

    # Diagnostic artifacts consumed by clustering-specific visualizations:
    # the elbow/knee curve (list of {n_clusters, score} points swept
    # during tuning) and, only when Hierarchical/Agglomerative was chosen,
    # the linkage matrix needed to render a dendrogram. Both are produced
    # during model_selection/tuning but only USED later during
    # visualization — stored in state so that later stage doesn't need to
    # refit anything to get them.
    clustering_elbow_curve: list[dict] | None
    clustering_linkage_matrix: list | None

    # reporting
    final_report: dict | None

    # Task Management
    current_task: str | None
    remaining_tasks: list[str]
    completed_tasks: list[str]
    execution_logs: list[dict]
