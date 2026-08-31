import json
import os
from datetime import datetime


class ReportingService:
    """Deterministic report builder. Aggregates all pipeline results."""

    def generate_report(self, state: dict) -> dict:
        report = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "user_query": state.get("user_query") or "N/A",
            "dataset_id": state.get("dataset_id") or "N/A",
            "target_column": state.get("target_column") or "N/A",
            "problem_type": self._get_problem_type(state),
            "pipeline_status": {
                "completed_tasks": state.get("completed_tasks") or [],
                "total_tasks": len(state.get("completed_tasks") or []),
            },
            "dataset_overview": self._build_dataset_overview(state),
            "cleaning": state.get("cleaning_report"),
            "eda": state.get("eda_report"),
            "visualization": {
                "num_charts": len(state.get("visualization_results") or []),
                "charts": state.get("visualization_results") or [],
            },
            "feature_engineering": state.get("feature_engineering_report"),
            "training": state.get("training_report"),
            "evaluation": state.get("evaluation_report"),
            "conclusions": self._build_conclusions(state),
        }

        # Save to disk
        os.makedirs("outputs/reports", exist_ok=True)
        report_path = f"outputs/reports/{state.get('dataset_id', 'report')}_final_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        report["report_path"] = report_path
        return report

    def _get_problem_type(self, state: dict) -> str:
        plan = state.get("analysis_plan")
        if plan:
            return getattr(plan, "problem_type", "unknown")
        return "unknown"

    def _build_dataset_overview(self, state: dict) -> dict:
        df = state.get("dataframe")
        if df is not None:
            return {
                "final_shape": list(df.shape),
                "columns": list(df.columns),
            }
        return {}

    def _build_conclusions(self, state: dict) -> dict:
        """Auto-generate key takeaways from training + evaluation."""
        training = state.get("training_report", {})
        evaluation = state.get("evaluation_report", {})

        conclusions = {
            "best_model": training.get("best_algorithm", "N/A"),
            "cv_score": training.get("best_mean_cv_score", "N/A"),
            "final_accuracy": None,
            "final_f1": None,
            "recommendation": "N/A",
        }

        if evaluation and evaluation.get("problem_type") == "classification":
            metrics = evaluation.get("metrics", {})
            conclusions["final_accuracy"] = metrics.get("accuracy")
            conclusions["final_f1"] = metrics.get("f1_weighted")

            acc = conclusions["final_accuracy"]
            if acc is not None:
                if acc >= 0.95:
                    conclusions["recommendation"] = "Excellent model performance. Ready for deployment."
                elif acc >= 0.90:
                    conclusions["recommendation"] = "Strong model performance. Consider hyperparameter tuning for marginal gains."
                elif acc >= 0.80:
                    conclusions["recommendation"] = "Good baseline. Explore feature engineering or ensemble methods."
                else:
                    conclusions["recommendation"] = "Performance needs improvement. Investigate data quality or model complexity."

        elif evaluation and evaluation.get("problem_type") == "regression":
            metrics = evaluation.get("metrics", {})
            r2 = metrics.get("r2")
            conclusions["final_r2"] = r2
            if r2 is not None:
                if r2 >= 0.90:
                    conclusions["recommendation"] = "Excellent fit. Model explains most variance."
                elif r2 >= 0.70:
                    conclusions["recommendation"] = "Reasonable fit. Consider non-linear models."
                else:
                    conclusions["recommendation"] = "Weak fit. Review features or collect more data."

        return conclusions
