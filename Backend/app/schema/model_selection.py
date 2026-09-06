from pydantic import BaseModel, Field
from typing import Literal, Optional


class ModelCandidate(BaseModel):
    """A single model the training service should try."""

    algorithm: Literal[
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
        "svm_rbf",
        "svm_linear",
        "neural_network_mlp",
    ] = Field(
        description="The ML algorithm to train and evaluate"
    )

    reason: str = Field(
        description="Clear justification for why THIS algorithm fits THIS specific dataset, citing evidence from EDA/FE reports"
    )

    estimated_time_seconds: int = Field(
        ge=1,
        description="Rough wall-clock estimate for training + CV on this dataset size"
    )

    hyperparams: dict = Field(
        default_factory=dict,
        description="Starting hyperparameters. Simple defaults only — no grid search at selection stage"
    )

    priority: int = Field(
        ge=1, le=5,
        description="Execution order. 1 = train first (fast baseline). Higher = more complex. Service trains in priority order."
    )


class ModelSelectionPlan(BaseModel):
    """The complete plan output by the LLM planner."""

    strategy: Literal["quick", "standard", "thorough"] = Field(
        description="Inferred from user tone and dataset complexity"
    )

    sample_size: Optional[int] = Field(
        default=None,
        description="If dataset rows > 100K, sample this many rows for model selection. None = use full dataset."
    )

    candidates: list[ModelCandidate] = Field(
        description="Ordered list of models to try. Usually 2-3, absolute maximum 4. Ordered by priority field."
    )

    cv_folds: int = Field(
        default=5,
        ge=2, le=10,
        description="Cross-validation folds. 3 for large datasets (>100K), 5 for small."
    )

    scoring_metric: Literal[
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f1_macro",
        "f1_weighted",
        "roc_auc",
        "neg_log_loss",
        "neg_mean_absolute_error",
        "neg_mean_squared_error",
        "neg_root_mean_squared_error",
        "r2",
        "explained_variance",
    ] = Field(
        description="Sklearn-compatible metric string. Classification: accuracy, f1, roc_auc, etc. Regression: neg_mean_absolute_error, r2, etc."
    )

    time_budget_minutes: int = Field(
        ge=1,
        description="Soft wall-clock limit. Training service stops if exceeded."
    )

    notes: list[str] = Field(
        default_factory=list,
        description="Why certain algorithms were excluded. Shows the LLM's reasoning."
    )
    forced_algorithm: Optional[Literal[
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
        "svm_rbf",
        "svm_linear",
        "neural_network_mlp",
    ]] = Field(
        default=None,
        description=(
            "Set ONLY if the user's query explicitly names a specific algorithm "
            "(e.g. 'use XGBoost', 'train a random forest'). When set, this algorithm "
            "is GUARANTEED to be the final selection regardless of CV score comparison "
            "against other candidates — it must still appear in `candidates`. "
            "Leave None for general 'find me the best model' queries where the "
            "algorithm choice is left to comparison."
        )
    )
