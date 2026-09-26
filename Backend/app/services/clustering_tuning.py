import os
import time
import logging
import joblib
import optuna
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage

from schema.clustering import ClusteringCandidate

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)


def _filter_noise(X, labels):
    """DBSCAN can label points as noise (-1) — not a real cluster.
    Clustering-quality metrics are only meaningful over actual cluster
    assignments, so noise points are excluded before scoring."""
    mask = labels != -1
    return X[mask], labels[mask]


class ClusteringTuningService:
    """Hyperparameter search for the WINNING clustering algorithm from
    model selection. Structurally different from HyperparameterTuningService
    in one key way: there's no cross_val_score to optimize, since there
    are no labels — instead each trial fits once on the full dataset and
    is scored by a clustering-quality metric (silhouette / davies_bouldin
    / calinski_harabasz). The thing being searched is also different: not
    a model's internal hyperparameters, but the STRUCTURAL parameter that
    determines cluster count/shape itself (n_clusters for kmeans/
    hierarchical, eps/min_samples for dbscan).

    Also produces two artifacts HyperparameterTuningService has no
    equivalent of: elbow_curve (score at each swept n_clusters value, for
    the elbow/knee visualization) and — only for hierarchical — the
    linkage matrix needed to render a dendrogram.
    """

    def tune(self, df: pd.DataFrame, best_candidate: ClusteringCandidate,
             scoring_metric: str, dataset_id: str,
             max_trials: int = 20, time_budget_seconds: int = 120):
        algorithm = best_candidate.algorithm

        logger.info(
            f"Starting clustering tuning: algorithm={algorithm}, "
            f"metric={scoring_metric}, max_trials={max_trials}, "
            f"time_budget={time_budget_seconds}s"
        )

        X_df = df.select_dtypes(include=["number", "bool"])
        non_numeric_cols = [c for c in df.columns if c not in X_df.columns]
        if non_numeric_cols:
            logger.warning(
                f"Dropping {len(non_numeric_cols)} non-numeric column(s) "
                f"before tuning: {non_numeric_cols}"
            )
        X = X_df.values

        study = optuna.create_study(direction="maximize")
        start_time = time.time()

        elbow_curve = []
        trial_records = []

        def objective(trial):
            params = self._suggest_params(trial, algorithm, n_rows=X.shape[0])
            model = self._build_model(algorithm, params)

            try:
                labels = model.fit_predict(X)
            except Exception as e:
                logger.warning(f"Trial failed to fit ({params}): {e}")
                raise optuna.TrialPruned()

            X_filtered, labels_filtered = _filter_noise(X, labels)
            n_real_clusters = len(set(labels_filtered))

            if n_real_clusters < 2:
                raise optuna.TrialPruned()

            internal_score, reported_score = self._score(
                X_filtered, labels_filtered, scoring_metric
            )

            trial_records.append({
                "params": params,
                "score": reported_score,
                "n_clusters_found": n_real_clusters,
            })

            # Elbow curve only makes sense against n_clusters directly —
            # dbscan has no such parameter, so it's excluded from this
            # curve rather than plotted against something meaningless.
            if "n_clusters" in params:
                elbow_curve.append({
                    "n_clusters": params["n_clusters"],
                    "score": reported_score,
                    "inertia": float(model.inertia_) if hasattr(model, "inertia_") else None,
                })

            return internal_score

        study.optimize(objective, n_trials=max_trials,
                       timeout=time_budget_seconds, show_progress_bar=False)
        elapsed = time.time() - start_time

        if not trial_records:
            raise RuntimeError(
                f"All tuning trials for '{algorithm}' failed or produced "
                f"degenerate clusterings (fewer than 2 real clusters)."
            )

        best_params = study.best_params
        final_model = self._build_model(algorithm, best_params)
        final_labels = final_model.fit_predict(X)

        n_clusters_final = len(set(final_labels) - {-1})
        n_noise_final = int((final_labels == -1).sum())

        linkage_matrix = None
        if algorithm == "hierarchical":
            try:
                linkage_matrix = linkage(X, method="ward").tolist()
            except Exception as e:
                logger.warning(f"Could not compute linkage matrix: {e}")

        os.makedirs("outputs/models", exist_ok=True)
        tuned_path = f"outputs/models/{dataset_id}_tuned_clustering_model.pkl"
        joblib.dump(final_model, tuned_path)

        report = {
            "original_algorithm": algorithm,
            "scoring_metric": scoring_metric,
            "best_trial_score": round(study.best_value, 5) if study.best_trial else None,
            "best_params": best_params,
            "num_trials_completed": len(study.trials),
            "n_clusters_found": n_clusters_final,
            "n_noise_points": n_noise_final,
            "time_seconds": round(elapsed, 2),
            "tuned_model_path": tuned_path,
            "elbow_curve": sorted(elbow_curve, key=lambda p: p["n_clusters"]) if elbow_curve else [],
            "cluster_labels": final_labels.tolist(),
            "linkage_matrix": linkage_matrix,
        }

        try:
            importances = optuna.importance.get_param_importances(study)
            report["param_importance"] = {
                k: round(float(v), 4) for k, v in importances.items()}
        except Exception as e:
            logger.warning(f"Could not compute param importances: {e}")
            report["param_importance"] = {}

        logger.info(
            f"Clustering tuning done: {report['num_trials_completed']} trials "
            f"in {report['time_seconds']}s, best_score={report['best_trial_score']}, "
            f"n_clusters={n_clusters_final}"
        )

        return final_model, report

    def _suggest_params(self, trial, algorithm: str, n_rows: int) -> dict:
        if algorithm in ("kmeans", "hierarchical"):
            max_k = min(15, max(2, n_rows // 2))
            return {"n_clusters": trial.suggest_int("n_clusters", 2, max_k)}

        if algorithm == "dbscan":
            return {
                "eps": trial.suggest_float("eps", 0.05, 2.0, log=True),
                "min_samples": trial.suggest_int("min_samples", 2, 20),
            }

        raise ValueError(f"Unknown clustering algorithm: {algorithm}")

    def _build_model(self, algorithm: str, params: dict):
        if algorithm == "kmeans":
            return KMeans(n_clusters=params["n_clusters"], random_state=42, n_init=10)
        if algorithm == "hierarchical":
            return AgglomerativeClustering(n_clusters=params["n_clusters"])
        if algorithm == "dbscan":
            return DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")

    def _score(self, X_filtered, labels_filtered, metric: str):
        """Returns (internal_score always-higher-is-better, reported_score
        in the metric's natural units) — same convention as
        ClusteringTrainingService._score."""
        if metric == "silhouette":
            score = float(silhouette_score(X_filtered, labels_filtered))
            return score, score
        if metric == "davies_bouldin":
            score = float(davies_bouldin_score(X_filtered, labels_filtered))
            return -score, score
        if metric == "calinski_harabasz":
            score = float(calinski_harabasz_score(X_filtered, labels_filtered))
            return score, score
        raise ValueError(f"Unknown clustering scoring metric: {metric}")
