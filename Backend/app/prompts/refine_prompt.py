REFINE_TARGET_PROMPT = """A user has already completed a full data analysis pipeline and now wants to change or redo one specific part of it.

Given their request, identify:
1. Which single pipeline stage it refers to: cleaning, eda, visualization, feature_engineering, model_selection, hyperparameter_tuning, evaluation, or reporting.
2. A clear restatement of the change they want, to hand to that stage's planner.
3. Your confidence (0.0 to 1.0) in the stage identification.

CRITICAL RULE: if the request changes, names, or switches WHICH algorithm/model to use — even if it also mentions tuning in the same sentence — the correct stage is ALWAYS model_selection, never hyperparameter_tuning. Changing the algorithm requires model_selection to rerun first; tuning of the new algorithm happens automatically afterward as part of the cascade, so you never need to (and must not) route directly to hyperparameter_tuning just because tuning was mentioned. Only route to hyperparameter_tuning when the algorithm itself is staying the same and only the tuning strategy/budget is changing (e.g. "tune it more aggressively", "run more trials", "try a wider search").

Examples:
- "add a pairplot to the EDA" -> stage: visualization, instruction: "add a pairplot chart"
- "try XGBoost instead" -> stage: model_selection, instruction: "prioritize xgboost as a candidate algorithm"
- "use a linear SVM and tune it" -> stage: model_selection, instruction: "switch to a linear SVM as the algorithm" (NOT hyperparameter_tuning — the algorithm is changing, so model_selection must rerun; tuning of the new model happens automatically via the cascade)
- "switch to random forest and tune its hyperparameters carefully" -> stage: model_selection, instruction: "switch to random forest as the algorithm"
- "redo cleaning, don't fill missing values with the median" -> stage: cleaning, instruction: "handle missing values differently, not with median fill"
- "run more hyperparameter tuning trials on the current model" -> stage: hyperparameter_tuning, instruction: "increase the number of tuning trials" (the algorithm itself isn't changing here)
- "tune it more aggressively" -> stage: hyperparameter_tuning, instruction: "use a more aggressive/thorough tuning strategy" (no algorithm named, so nothing to reselect)

Be conservative — if the request could reasonably mean two different stages, lower your confidence.
"""
