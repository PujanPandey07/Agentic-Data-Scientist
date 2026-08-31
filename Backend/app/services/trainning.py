import time
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.base import clone

from schema.model_selection import ModelSelectionPlan, ModelCandidate
import os
import joblib


class TrainingService:
    """Deterministic model trainer. Executes the ModelSelectionPlan sequentially."""

    def train(self, df: pd.DataFrame, plan: ModelSelectionPlan, target_column: str, dataset_id: str):
        """
        Train candidate models according to the plan.
        Returns: (best_trained_model, training_report, best_candidate)
        """
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in dataframe")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Determine if classification or regression from metric name
        is_classification = self._is_classification_metric(plan.scoring_metric)

        # Encode target if classification and string labels
        y_encoded = self._encode_target(y, is_classification)

        # Sample if needed
        X_sample, y_sample, used_sampling = self._sample_data(
            X, y_encoded, plan.sample_size, is_classification
        )

        # Train candidates sequentially
        results = []
        best_score = -float("inf")
        best_candidate = None
        best_model = None
        time_spent = 0.0

        # Sort by priority
        candidates = sorted(plan.candidates, key=lambda c: c.priority)

        for candidate in candidates:
            # Check time budget
            if time_spent >= plan.time_budget_minutes * 60:
                results.append({
                    "algorithm": candidate.algorithm,
                    "status": "skipped",
                    "reason": "Time budget exceeded",
                })
                continue

            start_time = time.time()

            try:
                # Get model instance
                model = self._get_model(candidate, is_classification)

                # Cross-validation
                scores = cross_val_score(
                    model,
                    X_sample,
                    y_sample,
                    cv=plan.cv_folds,
                    scoring=plan.scoring_metric,
                )
                mean_score = float(scores.mean())
                std_score = float(scores.std())

                elapsed = time.time() - start_time
                time_spent += elapsed

                result = {
                    "algorithm": candidate.algorithm,
                    "status": "success",
                    "mean_cv_score": mean_score,
                    "std_cv_score": std_score,
                    "cv_scores": scores.tolist(),
                    "estimated_time_seconds": candidate.estimated_time_seconds,
                    "actual_time_seconds": round(elapsed, 2),
                    "hyperparams": candidate.hyperparams,
                    "reason": candidate.reason,
                }
                results.append(result)

                # Track best
                if mean_score > best_score:
                    best_score = mean_score
                    best_candidate = candidate
                    best_model = model

            except Exception as e:
                elapsed = time.time() - start_time
                time_spent += elapsed

                results.append({
                    "algorithm": candidate.algorithm,
                    "status": "failed",
                    "error": str(e),
                    "actual_time_seconds": round(elapsed, 2),
                })

        # If no model succeeded, raise
        if best_model is None:
            raise RuntimeError(
                "All candidate models failed. Check logs for details.")

        # Retrain best model on FULL data
        if used_sampling:
            final_model = clone(best_model)
            final_model.fit(X, y_encoded)
            retrain_note = "Best model retrained on full dataset after selection on sample."
        else:
            final_model = clone(best_model)
            final_model.fit(X, y_encoded)
            retrain_note = "Best model trained on full dataset (no sampling needed)."

        # BUILD REPORT FIRST
        report = {
            "strategy": plan.strategy,
            "sample_size": plan.sample_size,
            "used_sampling": used_sampling,
            "full_dataset_shape": X.shape,
            "sample_shape": X_sample.shape if used_sampling else X.shape,
            "cv_folds": plan.cv_folds,
            "scoring_metric": plan.scoring_metric,
            "time_budget_minutes": plan.time_budget_minutes,
            "time_spent_seconds": round(time_spent, 2),
            "candidates_results": results,
            "best_algorithm": best_candidate.algorithm,
            "best_mean_cv_score": round(best_score, 5),
            "best_hyperparams": best_candidate.hyperparams,
            "best_reason": best_candidate.reason,
            "retrain_note": retrain_note,
            "notes": plan.notes,
        }

        # THEN save model and add path
        os.makedirs("outputs/models", exist_ok=True)
        model_path = f"outputs/models/{dataset_id}_best_model.pkl"
        joblib.dump(final_model, model_path)
        report["model_path"] = model_path

        return final_model, report, best_candidate

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    def _is_classification_metric(self, metric: str) -> bool:
        """Infer problem type from metric name."""
        classification_metrics = {
            "accuracy", "precision", "recall", "f1", "f1_macro",
            "f1_weighted", "roc_auc", "neg_log_loss",
        }
        return metric in classification_metrics

    def _encode_target(self, y: pd.Series, is_classification: bool) -> np.ndarray:
        """Encode string targets to integers for sklearn."""
        if not is_classification:
            return y.values

        if pd.api.types.is_numeric_dtype(y):
            return y.values

        # String labels → integer encoding
        le = LabelEncoder()
        return le.fit_transform(y.values)

    def _sample_data(self, X, y, sample_size, is_classification):
        """Sample data if needed. Stratified for classification, random for regression."""
        if sample_size is None or len(X) <= sample_size:
            return X, y, False

        if is_classification:
            # Stratified sampling preserves class balance
            X_sample, _, y_sample, _ = train_test_split(
                X, y,
                train_size=sample_size,
                stratify=y,
                random_state=42,
            )
        else:
            # Regression: stratify by target bins
            y_binned = pd.qcut(pd.Series(y), q=10,
                               labels=False, duplicates="drop")
            X_sample, _, y_sample, _ = train_test_split(
                X, y,
                train_size=sample_size,
                stratify=y_binned,
                random_state=42,
            )

        return X_sample, y_sample, True

    def _get_model(self, candidate: ModelCandidate, is_classification: bool):
        """Instantiate sklearn model based on algorithm name."""
        algo = candidate.algorithm
        params = candidate.hyperparams or {}

        # Classification models
        if is_classification:
            if algo == "logistic_regression":
                safe_params = {k: v for k, v in params.items() if k not in [
                    "multi_class"]}
                return LogisticRegression(**safe_params)
            elif algo == "random_forest":
                return RandomForestClassifier(**params)
            elif algo == "xgboost":
                try:
                    from xgboost import XGBClassifier
                    return XGBClassifier(use_label_encoder=False, **params)
                except ImportError:
                    raise ImportError(
                        "xgboost not installed. Run: pip install xgboost")
            elif algo == "lightgbm":
                try:
                    from lightgbm import LGBMClassifier
                    return LGBMClassifier(verbose=-1, **params)
                except ImportError:
                    raise ImportError(
                        "lightgbm not installed. Run: pip install lightgbm")
            elif algo == "svm_rbf":
                return SVC(kernel="rbf", **params)
            elif algo == "svm_linear":
                return SVC(kernel="linear", **params)
            elif algo == "neural_network_mlp":
                return MLPClassifier(**params)

        # Regression models
        else:
            if algo == "logistic_regression":
                raise ValueError(
                    "logistic_regression is for classification only")
            elif algo == "random_forest":
                return RandomForestRegressor(**params)
            elif algo == "xgboost":
                try:
                    from xgboost import XGBRegressor
                    return XGBRegressor(**params)
                except ImportError:
                    raise ImportError(
                        "xgboost not installed. Run: pip install xgboost")
            elif algo == "lightgbm":
                try:
                    from lightgbm import LGBMRegressor
                    return LGBMRegressor(verbose=-1, **params)
                except ImportError:
                    raise ImportError(
                        "lightgbm not installed. Run: pip install lightgbm")
            elif algo == "svm_rbf":
                return SVR(kernel="rbf", **params)
            elif algo == "svm_linear":
                return SVR(kernel="linear", **params)
            elif algo == "neural_network_mlp":
                return MLPRegressor(**params)

        raise ValueError(f"Unknown algorithm: {algo}")
