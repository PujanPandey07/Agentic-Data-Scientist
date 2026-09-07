# schema/constraints.py
from pydantic import BaseModel, Field
from typing import Literal


class ConstraintItem(BaseModel):
    stage: Literal[
        "cleaning", "eda", "visualization", "feature_engineering",
        "model_selection", "hyperparameter_tuning", "evaluation", "reporting"
    ]
    instruction: str = Field(
        description="The explicit constraint, restated clearly")


class UserConstraints(BaseModel):
    constraints: list[ConstraintItem] = Field(
        default_factory=list,
        description="Explicit, non-negotiable instructions the user stated. "
                    "Do NOT include vague preferences or things left to judgment."
    )


ALGORITHM_KEYWORDS = {
    "xgboost": "xgboost",
    "xgb": "xgboost",
    "random forest": "random_forest",
    "randomforest": "random_forest",
    "logistic regression": "logistic_regression",
    "logisticregression": "logistic_regression",
    "lightgbm": "lightgbm",
    "light gbm": "lightgbm",
    "lgbm": "lightgbm",
    "svm": "svm_rbf",
    "support vector machine": "svm_rbf",
    "support vector": "svm_rbf",
    "neural network": "neural_network_mlp",
    "mlp": "neural_network_mlp",
}


def detect_forced_algorithm(constraints: list[str] | None) -> str | None:
    """Deterministically check extracted model_selection constraints for an
    explicitly named algorithm. Returns the matching Literal value used by
    ModelCandidate/ModelSelectionPlan, or None if nothing matches."""
    if not constraints:
        return None

    combined = " ".join(constraints).lower()
    for keyword, algo in ALGORITHM_KEYWORDS.items():
        if keyword in combined:
            return algo

    return None
