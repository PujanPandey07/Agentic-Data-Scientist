def format_constraints(constraints: list[str] | None) -> str:
    """Format a stage's extracted constraints into an injectable prompt block.
    Returns an empty string if there are none — safe to always call."""
    if not constraints:
        return ""
    lines = "\n".join(f"- {c}" for c in constraints)
    return (
        f"\n\nMANDATORY CONSTRAINTS (must follow exactly — do not deviate "
        f"from these even if your own judgment would otherwise choose "
        f"differently):\n{lines}"
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
