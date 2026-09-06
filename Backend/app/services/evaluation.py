from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score,
)
import seaborn as sns
import matplotlib.pyplot as plt
import os
import logging
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")

logger = logging.getLogger(__name__)


class EvaluationService:
    def evaluate(self, df, target_column, model_path, problem_type):
        logger.info(
            f"Starting evaluation: model={model_path}, problem_type={problem_type}")

        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = joblib.load(model_path)

        if target_column not in df.columns:
            logger.error(f"Target column '{target_column}' not found")
            raise ValueError(f"Target column '{target_column}' not found")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        y_encoded = y
        if problem_type == "classification" and not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            logger.info(
                f"Label-encoded target '{target_column}' for classification")

        y_pred = model.predict(X)

        if problem_type == "classification":
            metrics = self._classification_metrics(y_encoded, y_pred)
            artifacts = self._classification_artifacts(
                y_encoded, y_pred, target_column)
        else:
            metrics = self._regression_metrics(y_encoded, y_pred)
            artifacts = self._regression_artifacts(
                y_encoded, y_pred, target_column)

        # Learning curve for ALL models
        lc_path = self._plot_learning_curve(model, X, y_encoded, problem_type)
        if lc_path:
            artifacts["learning_curve_path"] = lc_path

        # Boosting curve for tree boosters
        bc_path = self._plot_boosting_curve(model)
        if bc_path:
            artifacts["boosting_curve_path"] = bc_path

        logger.info(f"Evaluation done: {metrics}")

        return {
            "problem_type": problem_type,
            "num_samples": len(df),
            "num_features": X.shape[1],
            "metrics": metrics,
            "artifacts": artifacts,
        }

    def _plot_learning_curve(self, model, X, y, problem_type):
        os.makedirs("outputs/evaluation", exist_ok=True)
        try:
            scoring = "accuracy" if problem_type == "classification" else "neg_mean_squared_error"
            train_sizes, train_scores, val_scores = learning_curve(
                model, X, y, train_sizes=np.linspace(0.1, 1.0, 10),
                cv=5, scoring=scoring, n_jobs=-1, random_state=42,
            )
            train_mean, train_std = np.mean(
                train_scores, axis=1), np.std(train_scores, axis=1)
            val_mean, val_std = np.mean(
                val_scores, axis=1), np.std(val_scores, axis=1)

            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes, train_mean, 'o-',
                     color='blue', label='Training')
            plt.fill_between(train_sizes, train_mean - train_std,
                             train_mean + train_std, alpha=0.1, color='blue')
            plt.plot(train_sizes, val_mean, 'o-',
                     color='green', label='Validation')
            plt.fill_between(train_sizes, val_mean - val_std,
                             val_mean + val_std, alpha=0.1, color='green')
            plt.title(f"Learning Curve — {model.__class__.__name__}")
            plt.xlabel("Training Set Size")
            plt.ylabel("Score")
            plt.legend(loc="best")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            path = "outputs/evaluation/learning_curve.png"
            plt.savefig(path, dpi=150)
            plt.close()
            return path
        except Exception as e:
            logger.warning(f"Learning curve failed: {e}")
            return None

    def _plot_boosting_curve(self, model):
        """Only for models trained with eval_set (XGBoost)."""
        os.makedirs("outputs/evaluation", exist_ok=True)
        try:
            evals = getattr(model, 'evals_result_', None)
            if not evals:
                return None

            plt.figure(figsize=(10, 6))
            for dataset, metrics in evals.items():
                for metric, values in metrics.items():
                    plt.plot(values, label=f"{dataset} — {metric}")

            plt.title(f"Boosting Curve — {model.__class__.__name__}")
            plt.xlabel("Boosting Round")
            plt.ylabel("Metric")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            path = "outputs/evaluation/boosting_curve.png"
            plt.savefig(path, dpi=150)
            plt.close()
            return path
        except Exception as e:
            logger.warning(f"Boosting curve failed: {e}")
            return None

    def _classification_metrics(self, y_true, y_pred):
        return {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 5),
            "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 5),
            "precision_weighted": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 5),
            "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 5),
            "recall_weighted": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 5),
            "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 5),
            "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 5),
        }

    def _classification_artifacts(self, y_true, y_pred, target_name):
        os.makedirs("outputs/evaluation", exist_ok=True)
        cm = confusion_matrix(y_true, y_pred)
        labels = sorted(list(set(y_true) | set(y_pred)))

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.title(f"Confusion Matrix — {target_name}")
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.tight_layout()

        path = "outputs/evaluation/confusion_matrix.png"
        plt.savefig(path, dpi=150)
        plt.close()

        return {
            "confusion_matrix_path": path,
            "confusion_matrix_values": cm.tolist(),
            "labels": labels,
        }

    def _regression_metrics(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        return {
            "mae": round(float(mae), 5),
            "mse": round(float(mse), 5),
            "rmse": round(float(np.sqrt(mse)), 5),
            "r2": round(float(r2_score(y_true, y_pred)), 5),
        }

    def _regression_artifacts(self, y_true, y_pred, target_name):
        os.makedirs("outputs/evaluation", exist_ok=True)
        residuals = np.array(y_true) - np.array(y_pred)

        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.6, edgecolors="k")
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("Predicted")
        plt.ylabel("Residual")
        plt.title(f"Residual Plot — {target_name}")
        plt.tight_layout()
        path = "outputs/evaluation/residual_plot.png"
        plt.savefig(path, dpi=150)
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolors="k")
        min_v, max_v = min(min(y_true), min(y_pred)), max(
            max(y_true), max(y_pred))
        plt.plot([min_v, max_v], [min_v, max_v], "r--", lw=2)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"Actual vs Predicted — {target_name}")
        plt.tight_layout()
        path2 = "outputs/evaluation/actual_vs_predicted.png"
        plt.savefig(path2, dpi=150)
        plt.close()

        return {
            "residual_plot_path": path,
            "actual_vs_predicted_path": path2,
        }
