from pydantic import BaseModel, Field
from typing import Literal, Optional

CLUSTERING_ALGORITHM_NAMES = Literal[
    "kmeans",
    "dbscan",
    "hierarchical",
]


class ClusteringCandidate(BaseModel):
    """A single clustering algorithm the training service should try."""

    algorithm: CLUSTERING_ALGORITHM_NAMES = Field(
        description="The clustering algorithm to fit and score"
    )

    reason: str = Field(
        description="Clear justification for why THIS algorithm fits THIS specific dataset, citing evidence from EDA/FE reports"
    )

    estimated_time_seconds: int = Field(
        ge=1,
        description="Rough wall-clock estimate for fitting this algorithm on this dataset size"
    )

    hyperparams: dict = Field(
        default_factory=dict,
        description=(
            "Starting hyperparameters. For kmeans/hierarchical this is "
            "usually {'n_clusters': N}; for dbscan it's {'eps': E, "
            "'min_samples': M}. Simple defaults only — the hyperparameter "
            "tuning stage searches this space properly."
        )
    )

    priority: int = Field(
        ge=1, le=5,
        description="Execution order. 1 = fit first (fastest/simplest). Service fits in priority order."
    )


class ClusteringPlan(BaseModel):
    """The complete clustering plan output by the LLM planner."""

    strategy: Literal["quick", "standard", "thorough"] = Field(
        description="Inferred from user tone and dataset complexity"
    )

    candidates: list[ClusteringCandidate] = Field(
        description="Ordered list of clustering algorithms to try. Usually 1-3, absolute maximum 4."
    )

    scoring_metric: Literal[
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
    ] = Field(
        default="silhouette",
        description=(
            "Clustering-quality metric to compare candidates on. "
            "silhouette and calinski_harabasz: higher is better. "
            "davies_bouldin: lower is better — the training service "
            "handles the sign internally, this field just names which "
            "metric the user's plan is being judged on."
        )
    )

    time_budget_minutes: int = Field(
        ge=1,
        description="Soft wall-clock limit. Training service stops if exceeded."
    )

    notes: list[str] = Field(
        default_factory=list,
        description="Why certain algorithms were excluded. Shows the LLM's reasoning."
    )

    forced_algorithm: Optional[CLUSTERING_ALGORITHM_NAMES] = Field(
        default=None,
        description=(
            "Set ONLY if the user's query explicitly names a specific "
            "algorithm (e.g. 'use DBSCAN', 'cluster with KMeans'). When "
            "set, this algorithm is GUARANTEED to be the final selection "
            "regardless of score comparison against other candidates — it "
            "must still appear in `candidates`. Leave None for general "
            "'find natural groupings' queries where the algorithm choice "
            "is left to comparison."
        )
    )
