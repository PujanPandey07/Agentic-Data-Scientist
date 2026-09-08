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


# Order matters: this dict is scanned top-to-bottom and the FIRST matching
# keyword wins. More specific multi-word phrases must come before shorter
# substrings they contain — e.g. "linear svm" must be checked before the
# bare "svm" entry, since "svm" is itself a substring of "linear svm" and
# would otherwise always win first regardless of which phrase is actually
# more specific to the user's request.
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

    # Linear SVM variants — MUST precede the generic "svm"/"support vector"
    # entries below.
    "linear svm": "svm_linear",
    "svm linear": "svm_linear",
    "linear support vector machine": "svm_linear",
    "linear support vector": "svm_linear",
    "svm with a linear kernel": "svm_linear",
    "linear kernel svm": "svm_linear",

    # RBF SVM — explicit variants, plus the bare/generic fallback.
    "rbf svm": "svm_rbf",
    "svm rbf": "svm_rbf",
    "svm with an rbf kernel": "svm_rbf",
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
