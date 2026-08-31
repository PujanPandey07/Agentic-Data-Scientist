import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
)
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend


class EvaluationService:
    """Deterministic evaluation. Loads trained model, computes metrics, generates artifacts."""

    def evaluate(self, df: pd.DataFrame, target_column: str, model_path: str, problem_type: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        # Load model
        model = joblib.load(model_path)

        # Prepare data
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Predict
        y_pred = model.predict(X)

        # FIX: Model predicts encoded integers, but y_true might still be strings
        # Encode y to match the model's output format
        if problem_type == "classification" and not pd.api.types.is_numeric_dtype(y):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Compute metrics based on problem type
        if problem_type == "classification":
            metrics = self._classification_metrics(y, y_pred)
            artifacts = self._classification_artifacts(
                y, y_pred, target_column)
        elif problem_type == "regression":
            metrics = self._regression_metrics(y, y_pred)
            artifacts = self._regression_artifacts(y, y_pred, target_column)
        else:
            raise ValueError(f"Unknown problem_type: {problem_type}")

        return {
            "problem_type": problem_type,
            "num_samples": len(df),
            "num_features": X.shape[1],
            "metrics": metrics,
            "artifacts": artifacts,
        }

    # --------------------------------------------------------------------- #
    # Classification
    # --------------------------------------------------------------------- #

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

    def _classification_artifacts(self, y_true, y_pred, target_name: str):
        os.makedirs("outputs/evaluation", exist_ok=True)

        cm = confusion_matrix(y_true, y_pred)
        labels = sorted(list(set(y_true) | set(y_pred)))
        if pd.api.types.is_numeric_dtype(pd.Series(y_true)):
            # Already encoded, just use as-is for matrix
            pass

        cm = confusion_matrix(y_true, y_pred)
        labels = sorted(list(set(y_true) | set(y_pred)))

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.title(f"Confusion Matrix — {target_name}")
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.tight_layout()

        path = f"outputs/evaluation/confusion_matrix.png"
        plt.savefig(path, dpi=150)
        plt.close()

        return {
            "confusion_matrix_path": path,
            "confusion_matrix_values": cm.tolist(),
            "labels": labels,
        }

    # --------------------------------------------------------------------- #
    # Regression
    # --------------------------------------------------------------------- #

    def _regression_metrics(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        return {
            "mae": round(float(mae), 5),
            "mse": round(float(mse), 5),
            "rmse": round(float(rmse), 5),
            "r2": round(float(r2_score(y_true, y_pred)), 5),
        }

    def _regression_artifacts(self, y_true, y_pred, target_name: str):
        os.makedirs("outputs/evaluation", exist_ok=True)

        # Residual plot
        residuals = np.array(y_true) - np.array(y_pred)

        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.6, edgecolors="k")
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("Predicted")
        plt.ylabel("Residual")
        plt.title(f"Residual Plot — {target_name}")
        plt.tight_layout()

        path = f"outputs/evaluation/residual_plot.png"
        plt.savefig(path, dpi=150)
        plt.close()

        # Actual vs Predicted plot
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolors="k")
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"Actual vs Predicted — {target_name}")
        plt.tight_layout()

        path2 = f"outputs/evaluation/actual_vs_predicted.png"
        plt.savefig(path2, dpi=150)
        plt.close()

        return {
            "residual_plot_path": path,
            "actual_vs_predicted_path": path2,
        }
