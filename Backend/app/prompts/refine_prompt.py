REFINE_TARGET_PROMPT = """A user has already completed a full data analysis pipeline and now wants to change or redo one specific part of it.

Given their request, identify:
1. Which single pipeline stage it refers to: cleaning, eda, visualization, feature_engineering, model_selection, hyperparameter_tuning, evaluation, or reporting.
2. A clear restatement of the change they want, to hand to that stage's planner.
3. Your confidence (0.0 to 1.0) in the stage identification.

Examples:
- "add a pairplot to the EDA" -> stage: visualization, instruction: "add a pairplot chart"
- "try XGBoost instead" -> stage: model_selection, instruction: "prioritize xgboost as a candidate algorithm"
- "redo cleaning, don't fill missing values with the median" -> stage: cleaning, instruction: "handle missing values differently, not with median fill"

Be conservative — if the request could reasonably mean two different stages, lower your confidence.
"""
