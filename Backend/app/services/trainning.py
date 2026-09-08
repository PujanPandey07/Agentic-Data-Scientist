import time
import logging
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

from schema.model_selection import ModelSelectionPlan, ModelCandidate
import os
import joblib

logger = logging.getLogger(__name__)


def _fit_with_convergence_check(fit_callable):
    """Run a fit/cross-validation call while watching for ConvergenceWarning.
    Returns (result, converged: bool). Used so training/tuning reports can
    tell explain_result whether a model actually finished training, instead
    of the LLM having to guess at causes for a CV-vs-test score gap that
    might really just be non-convergence, not overfitting."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        result = fit_callable()
    converged = not any(
        issubclass(w.category, ConvergenceWarning) for w in caught
    )
    return result, converged


class TrainingService:
    """Deterministic model trainer. Executes the ModelSelectionPlan sequentially."""

    def train(self, df: pd.DataFrame, plan: ModelSelectionPlan, target_column: str, dataset_id: str):
        """
        Train candidate models according to the plan.
        Returns: (best_trained_model, training_report, best_candidate)
        """
        if target_column not in df.columns:
            logger.error(
                f"Target column '{target_column}' not found in dataframe")
            raise ValueError(
                f"Target column '{target_column}' not found in dataframe")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        is_classification = self._is_classification_metric(plan.scoring_metric)

        logger.info(
            f"Starting training: {len(plan.candidates)} candidates, "
            f"classification={is_classification}, metric={plan.scoring_metric}"
        )

        y_encoded = self._encode_target(y, is_classification)

        X_sample, y_sample, used_sampling = self._sample_data(
            X, y_encoded, plan.sample_size, is_classification
        )
        if used_sampling:
            logger.info(
                f"Sampled data for candidate selection: {X_sample.shape}")

        results = []
        best_score = -float("inf")
        best_candidate = None
        best_model = None
        time_spent = 0.0

        candidates = sorted(plan.candidates, key=lambda c: c.priority)

        for candidate in candidates:
            if time_spent >= plan.time_budget_minutes * 60:
                results.append({
                    "algorithm": candidate.algorithm,
                    "status": "skipped",
                    "reason": "Time budget exceeded",
                })
                logger.warning(
                    f"Skipped '{candidate.algorithm}': time budget exceeded")
                continue

            start_time = time.time()

            try:
                model = self._get_model(candidate, is_classification)

                scores, converged = _fit_with_convergence_check(
                    lambda: cross_val_score(
                        model, X_sample, y_sample,
                        cv=plan.cv_folds, scoring=plan.scoring_metric,
                    )
                )
                mean_score = float(scores.mean())
                std_score = float(scores.std())

                elapsed = time.time() - start_time
                time_spent += elapsed

                if not converged:
                    logger.warning(
                        f"'{candidate.algorithm}' did not fully converge "
                        f"during cross-validation — score may be unstable"
                    )

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
                    "converged": converged,
                }
                results.append(result)
                logger.info(
                    f"'{candidate.algorithm}' scored {round(mean_score, 5)} "
                    f"(+/- {round(std_score, 5)}) in {round(elapsed, 2)}s "
                    f"[converged={converged}]"
                )

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
                logger.warning(f"'{candidate.algorithm}' failed: {e}")

        if best_model is None:
            logger.error("All candidate models failed")
            raise RuntimeError(
                "All candidate models failed. Check logs for details.")

        logger.info(
            f"Best candidate: {best_candidate.algorithm} (score={round(best_score, 5)})")

        forced_override_applied = False
        forced_algorithm = getattr(plan, "forced_algorithm", None)

        if forced_algorithm:
            forced_result = next(
                (r for r in results if r["algorithm"] ==
                 forced_algorithm and r["status"] == "success"),
                None,
            )
            if forced_result is not None:
                forced_candidate = next(
                    c for c in candidates if c.algorithm == forced_algorithm
                )
                if forced_algorithm != best_candidate.algorithm:
                    logger.info(
                        f"User explicitly requested '{forced_algorithm}' "
                        f"(scored {forced_result['mean_cv_score']}) — overriding "
                        f"CV-selected winner '{best_candidate.algorithm}' "
                        f"(scored {round(best_score, 5)})"
                    )
                best_candidate = forced_candidate
                best_score = forced_result["mean_cv_score"]
                best_model = self._get_model(
                    forced_candidate, is_classification)
                forced_override_applied = True
            else:
                logger.warning(
                    f"User requested '{forced_algorithm}' but it failed to train "
                    f"or wasn't among candidates — falling back to CV-selected "
                    f"winner '{best_candidate.algorithm}'"
                )

        # Retrain best model on FULL data — this is the model that actually
        # gets saved/evaluated/deployed, so its convergence status matters
        # most of all.
        final_model = clone(best_model)
        _, final_model_converged = _fit_with_convergence_check(
            lambda: final_model.fit(X, y_encoded)
        )

        if not final_model_converged:
            logger.warning(
                f"Final retrained '{best_candidate.algorithm}' model did NOT "
                f"fully converge — reported scores may be unstable/unreliable, "
                f"not just 'slightly overfit'"
            )

        retrain_note = (
            "Best model retrained on full dataset after selection on sample."
            if used_sampling else
            "Best model trained on full dataset (no sampling needed)."
        )

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
            "forced_algorithm": forced_algorithm,
            "forced_override_applied": forced_override_applied,
            "retrain_note": retrain_note,
            "final_model_converged": final_model_converged,
            "notes": plan.notes,
        }

        os.makedirs("outputs/models", exist_ok=True)
        model_path = f"outputs/models/{dataset_id}_best_model.pkl"
        joblib.dump(final_model, model_path)
        report["model_path"] = model_path

        logger.info(f"Training done, model saved to {model_path}")

        return final_model, report, best_candidate

    # --------------------------------------------------------------------- #
    # Helpers — unchanged from before
    # --------------------------------------------------------------------- #

    def _is_classification_metric(self, metric: str) -> bool:
        classification_metrics = {
            "accuracy", "precision", "recall", "f1", "f1_macro",
            "f1_weighted", "roc_auc", "neg_log_loss",
        }
        return metric in classification_metrics

    def _encode_target(self, y: pd.Series, is_classification: bool) -> np.ndarray:
        if not is_classification:
            return y.values
        if pd.api.types.is_numeric_dtype(y):
            return y.values
        le = LabelEncoder()
        return le.fit_transform(y.values)

    def _sample_data(self, X, y, sample_size, is_classification):
        if sample_size is None or len(X) <= sample_size:
            return X, y, False

        if is_classification:
            X_sample, _, y_sample, _ = train_test_split(
                X, y, train_size=sample_size, stratify=y, random_state=42,
            )
        else:
            y_binned = pd.qcut(pd.Series(y), q=10,
                               labels=False, duplicates="drop")
            X_sample, _, y_sample, _ = train_test_split(
                X, y, train_size=sample_size, stratify=y_binned, random_state=42,
            )

        return X_sample, y_sample, True

    def _get_model(self, candidate: ModelCandidate, is_classification: bool):
        algo = candidate.algorithm
        params = candidate.hyperparams or {}

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
                    safe_params = {
                        k: v for k, v in params.items() if k != "use_label_encoder"}
                    return XGBClassifier(**safe_params)
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
                params = {k: v for k, v in params.items() if k != "kernel"}
                return SVC(kernel="rbf", **params)
            elif algo == "svm_linear":
                params = {k: v for k, v in params.items() if k != "kernel"}
                return SVC(kernel="linear", **params)
            elif algo == "neural_network_mlp":
                return MLPClassifier(**params)

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
