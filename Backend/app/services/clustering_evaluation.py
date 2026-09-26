import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


class ClusteringEvaluationService:
    """Deterministic — no model fitting here. Purely scores the final
    cluster_labels (produced by tuning, or training if tuning was
    skipped) against the dataset used to produce them. Mirrors
    EvaluationService's role in the pipeline (a final, fast step after
    the modeling work is done) but with clustering-appropriate metrics
    instead of accuracy/F1/RMSE/R² — there's no ground truth here.
    """

    def evaluate(self, df: pd.DataFrame, labels: list[int], dataset_id: str) -> dict:
        X_df = df.select_dtypes(include=["number", "bool"])
        X = X_df.values
        labels_arr = np.array(labels)

        if len(labels_arr) != len(X):
            return {
                "error": (
                    f"Label count ({len(labels_arr)}) doesn't match dataset "
                    f"row count ({len(X)}) — dataset may have changed since "
                    f"clustering ran."
                )
            }

        mask = labels_arr != -1
        X_filtered = X[mask]
        labels_filtered = labels_arr[mask]
        n_real_clusters = len(set(labels_filtered))
        n_noise = int((labels_arr == -1).sum())

        cluster_sizes = {
            int(label): int((labels_arr == label).sum())
            for label in sorted(set(labels_arr))
            if label != -1
        }

        metrics = {}
        if n_real_clusters >= 2:
            metrics["silhouette_score"] = round(
                float(silhouette_score(X_filtered, labels_filtered)), 5)
            metrics["davies_bouldin_index"] = round(
                float(davies_bouldin_score(X_filtered, labels_filtered)), 5)
            metrics["calinski_harabasz_index"] = round(
                float(calinski_harabasz_score(X_filtered, labels_filtered)), 5)
        else:
            metrics["warning"] = (
                f"Only {n_real_clusters} real cluster(s) found after excluding "
                f"noise — quality metrics require at least 2 to be meaningful."
            )

        return {
            "n_clusters": n_real_clusters,
            "n_noise_points": n_noise,
            "cluster_sizes": cluster_sizes,
            "metrics": metrics,
            "dataset_id": dataset_id,
        }


clustering_evaluation_service = ClusteringEvaluationService()
