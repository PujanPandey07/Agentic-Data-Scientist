import time
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import os
import joblib

from schema.clustering import ClusteringPlan, ClusteringCandidate

logger = logging.getLogger(__name__)


def _filter_noise(X: np.ndarray, labels: np.ndarray):
    """DBSCAN can label points as noise (-1), which isn't a real cluster.
    Clustering-quality metrics (silhouette, Davies-Bouldin, Calinski-
    Harabasz) are only meaningful over actual cluster assignments, so
    noise points are excluded before scoring — the standard convention
    when evaluating DBSCAN results. Returns (X_filtered, labels_filtered).
    """
    mask = labels != -1
    return X[mask], labels[mask]


class ClusteringTrainingService:
    """Deterministic clustering trainer. Mirrors TrainingService's shape
    (sequential candidates in priority order, best-of comparison, forced
    algorithm override, save winning model to disk) but with two
    structural differences forced by the lack of labels:

    1. No cross-validation and no train/test split — each candidate is
       fit ONCE on the full (numeric) dataset. There's nothing to
       validate predictions against, so CV has no meaning here.
    2. Candidates can produce a DEGENERATE result (fewer than 2 real
       clusters, most commonly DBSCAN finding only noise or one blob) —
       every clustering-quality metric is undefined in that case, so
       such candidates are marked failed rather than silently scored.
    """

    def train(self, df: pd.DataFrame, plan: ClusteringPlan, dataset_id: str):
        """
        Fit candidate clustering algorithms according to the plan.
        Returns: (best_fitted_model, best_labels: list[int], training_report, best_candidate)
        """
        # Same defensive guard as TrainingService: feature engineering may
        # leave raw non-numeric columns alongside encoded counterparts.
        # Clustering algorithms need purely numeric input.
        X_df = df.select_dtypes(include=["number", "bool"]).copy()
        dropped_cols = [c for c in df.columns if c not in X_df.columns]
        if dropped_cols:
            logger.warning(
                f"Dropping {len(dropped_cols)} non-numeric column(s) before "
                f"clustering (likely raw versions of encoded columns): {dropped_cols}"
            )

        if X_df.shape[1] == 0:
            raise ValueError(
                "No numeric columns available for clustering after dropping "
                "non-numeric columns."
            )

        X = X_df.values

        logger.info(
            f"Starting clustering: {len(plan.candidates)} candidates, "
            f"metric={plan.scoring_metric}, shape={X.shape}"
        )

        results = []
        best_internal_score = -float("inf")
        best_reported_score = None
        best_candidate = None
        best_model = None
        best_labels = None
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
                model = self._get_model(candidate)
                labels = model.fit_predict(X)

                elapsed = time.time() - start_time
                time_spent += elapsed

                n_clusters_found = len(set(labels) - {-1})
                n_noise = int((labels == -1).sum())

                internal_score, reported_score = self._score(
                    X, labels, plan.scoring_metric
                )

                if internal_score is None:
                    results.append({
                        "algorithm": candidate.algorithm,
                        "status": "failed",
                        "error": (
                            f"Fewer than 2 real clusters found "
                            f"(n_clusters={n_clusters_found}, noise={n_noise}) "
                            f"— no clustering-quality metric is meaningful here."
                        ),
                        "actual_time_seconds": round(elapsed, 2),
                    })
                    logger.warning(
                        f"'{candidate.algorithm}' produced fewer than 2 real "
                        f"clusters — excluded from comparison"
                    )
                    continue

                result = {
                    "algorithm": candidate.algorithm,
                    "status": "success",
                    "score": round(reported_score, 5),
                    "scoring_metric": plan.scoring_metric,
                    "n_clusters_found": n_clusters_found,
                    "n_noise_points": n_noise,
                    "estimated_time_seconds": candidate.estimated_time_seconds,
                    "actual_time_seconds": round(elapsed, 2),
                    "hyperparams": candidate.hyperparams,
                    "reason": candidate.reason,
                }
                results.append(result)
                logger.info(
                    f"'{candidate.algorithm}' scored {round(reported_score, 5)} "
                    f"({plan.scoring_metric}), {n_clusters_found} clusters, "
                    f"{n_noise} noise points, in {round(elapsed, 2)}s"
                )

                if internal_score > best_internal_score:
                    best_internal_score = internal_score
                    best_reported_score = reported_score
                    best_candidate = candidate
                    best_model = model
                    best_labels = labels

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
            logger.error("All candidate clustering algorithms failed")
            raise RuntimeError(
                "All candidate clustering algorithms failed or produced "
                "degenerate clusterings (fewer than 2 real clusters). "
                "Check logs for details."
            )

        logger.info(
            f"Best candidate: {best_candidate.algorithm} "
            f"(score={round(best_reported_score, 5)})"
        )

        forced_override_applied = False
        forced_algorithm = getattr(plan, "forced_algorithm", None)

        if forced_algorithm:
            forced_result = next(
                (r for r in results if r["algorithm"] ==
                 forced_algorithm and r["status"] == "success"),
                None,
            )
            if forced_result is not None and forced_algorithm != best_candidate.algorithm:
                logger.info(
                    f"User explicitly requested '{forced_algorithm}' "
                    f"(scored {forced_result['score']}) — overriding "
                    f"metric-selected winner '{best_candidate.algorithm}' "
                    f"(scored {round(best_reported_score, 5)})"
                )
                forced_candidate = next(
                    c for c in candidates if c.algorithm == forced_algorithm
                )
                forced_model = self._get_model(forced_candidate)
                forced_labels = forced_model.fit_predict(X)
                best_candidate = forced_candidate
                best_model = forced_model
                best_labels = forced_labels
                best_reported_score = forced_result["score"]
                forced_override_applied = True
            elif forced_result is None:
                logger.warning(
                    f"User requested '{forced_algorithm}' but it failed or "
                    f"wasn't among successful candidates — falling back to "
                    f"metric-selected winner '{best_candidate.algorithm}'"
                )

        n_clusters_final = len(set(best_labels) - {-1})
        n_noise_final = int((best_labels == -1).sum())

        report = {
            "strategy": plan.strategy,
            "full_dataset_shape": X.shape,
            "scoring_metric": plan.scoring_metric,
            "time_budget_minutes": plan.time_budget_minutes,
            "time_spent_seconds": round(time_spent, 2),
            "candidates_results": results,
            "best_algorithm": best_candidate.algorithm,
            "best_score": round(best_reported_score, 5),
            "best_hyperparams": best_candidate.hyperparams,
            "best_reason": best_candidate.reason,
            "n_clusters_found": n_clusters_final,
            "n_noise_points": n_noise_final,
            "forced_algorithm": forced_algorithm,
            "forced_override_applied": forced_override_applied,
            "notes": plan.notes,
        }

        os.makedirs("outputs/models", exist_ok=True)
        model_path = f"outputs/models/{dataset_id}_best_clustering_model.pkl"
        joblib.dump(best_model, model_path)
        report["model_path"] = model_path

        logger.info(f"Clustering done, model saved to {model_path}")

        return best_model, best_labels.tolist(), report, best_candidate

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    def _score(self, X: np.ndarray, labels: np.ndarray, metric: str):
        """Returns (internal_score, reported_score).

        internal_score is ALWAYS higher-is-better, used only for comparing
        candidates against each other. reported_score is in the metric's
        natural units, for humans/reports to read — identical to
        internal_score for silhouette and calinski_harabasz, but negated
        for davies_bouldin (which is natively lower-is-better).

        Returns (None, None) if fewer than 2 real clusters remain after
        excluding noise — no clustering metric is defined in that case.
        """
        X_filtered, labels_filtered = _filter_noise(X, labels)
        n_real_clusters = len(set(labels_filtered))

        if n_real_clusters < 2:
            return None, None

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

    def _get_model(self, candidate: ClusteringCandidate):
        algo = candidate.algorithm
        params = dict(candidate.hyperparams or {})

        if algo == "kmeans":
            params.setdefault("n_clusters", 3)
            params.setdefault("random_state", 42)
            params.setdefault("n_init", 10)
            return KMeans(**params)

        if algo == "dbscan":
            params.setdefault("eps", 0.5)
            params.setdefault("min_samples", 5)
            return DBSCAN(**params)

        if algo == "hierarchical":
            params.setdefault("n_clusters", 3)
            return AgglomerativeClustering(**params)

        raise ValueError(f"Unknown clustering algorithm: {algo}")
