import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ReportingService:
    """Deterministic report builder. Aggregates results from stages that
    actually ran in THIS invocation — never stale data left over in the
    checkpoint from an earlier, unrelated session on the same thread."""

    def generate_report(self, state: dict) -> dict:
        logger.info(
            f"Generating final report for dataset {state.get('dataset_id')}")

        completed = state.get("completed_tasks") or []

        ran_feature_engineering = "feature_engineering" in completed
        ran_model_selection = "model_selection" in completed
        ran_hyperparameter_tuning = "hyperparameter_tuning" in completed
        ran_evaluation = "evaluation" in completed

        report = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "user_query": state.get("user_query") or "N/A",
            "dataset_id": state.get("dataset_id") or "N/A",
            "target_column": state.get("target_column") or "N/A",
            "problem_type": self._get_problem_type(state),
            "pipeline_status": {
                "completed_tasks": completed,
                "total_tasks": len(completed),
            },
            "dataset_overview": self._build_dataset_overview(state),
            "cleaning": state.get("cleaning_report"),
            "eda": state.get("eda_report"),
            "visualization": {
                "num_charts": len(state.get("visualization_results") or []),
                "charts": state.get("visualization_results") or [],
            },
            "feature_engineering": (
                state.get("feature_engineering_report")
                if ran_feature_engineering else None
            ),
            "training": (
                state.get("training_report")
                if ran_model_selection else None
            ),
            "hyperparameter_tuning": (
                state.get("hyperparameter_tuning_report")
                if ran_hyperparameter_tuning else None
            ),
            "evaluation": (
                state.get("evaluation_report")
                if ran_evaluation else None
            ),
            "conclusions": self._build_conclusions(state, ran_model_selection, ran_evaluation),
        }

        # Save to disk
        os.makedirs("outputs/reports", exist_ok=True)
        report_path = f"outputs/reports/{state.get('dataset_id', 'report')}_final_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        report["report_path"] = report_path

        logger.info(
            f"Report saved to {report_path} "
            f"(recommendation: {report['conclusions'].get('recommendation', 'N/A')})"
        )

        return report

    def _get_problem_type(self, state: dict) -> str:
        plan = state.get("analysis_plan")
        if plan:
            # getattr's default only kicks in when the attribute is missing,
            # not when it exists but is None — so fall back explicitly too.
            return getattr(plan, "problem_type", None) or "unknown"
        return "unknown"

    def _build_dataset_overview(self, state: dict) -> dict:
        df = state.get("dataframe")
        if df is not None:
            return {
                "final_shape": list(df.shape),
                "columns": list(df.columns),
            }
        return {}

    def _build_conclusions(
        self, state: dict, ran_model_selection: bool, ran_evaluation: bool
    ) -> dict:
        """Auto-generate key takeaways from training + evaluation — but
        only if those stages actually ran in this invocation. Otherwise
        this run made no claims about model quality, and the report
        shouldn't invent any."""

        if not ran_model_selection:
            return {
                "best_model": "N/A",
                "cv_score": "N/A",
                "final_accuracy": None,
                "final_f1": None,
                "recommendation": (
                    "No model training was part of this run — "
                    "see cleaning/EDA/visualization results above."
                ),
            }

        training = state.get("training_report") or {}
        tuning = state.get("hyperparameter_tuning_report") or {}

        # If either the initially-trained OR the tuned model failed to fully
        # converge, any recommendation about accuracy/overfitting is on
        # shaky ground — flag this explicitly instead of letting a
        # downstream reader (or explain_result) invent a wrong explanation
        # for a CV-vs-test score gap that's actually just non-convergence.
        convergence_warning = None
        if training.get("final_model_converged") is False:
            convergence_warning = (
                "WARNING: the trained model did not fully converge during "
                "training. Reported scores may be unstable and not "
                "representative of the model's true performance — this is "
                "a different and more serious issue than overfitting. "
                "Common causes: unscaled features (especially for SVM/"
                "neural network models) or too few iterations."
            )
        elif tuning.get("final_model_converged") is False:
            convergence_warning = (
                "WARNING: the tuned model did not fully converge during "
                "hyperparameter tuning. Reported scores may be unstable and "
                "not representative of the model's true performance — this "
                "is a different and more serious issue than overfitting. "
                "Common causes: unscaled features (especially for SVM/"
                "neural network models) or too few iterations."
            )

        conclusions = {
            "best_model": training.get("best_algorithm", "N/A"),
            "cv_score": training.get("best_mean_cv_score", "N/A"),
            "final_accuracy": None,
            "final_f1": None,
            "recommendation": "N/A",
        }

        if convergence_warning:
            conclusions["convergence_warning"] = convergence_warning

        if not ran_evaluation:
            conclusions["recommendation"] = (
                "Model was trained/selected, but evaluation did not run "
                "in this session."
            )
            if convergence_warning:
                conclusions["recommendation"] += f" {convergence_warning}"
            return conclusions

        evaluation = state.get("evaluation_report") or {}

        if evaluation.get("problem_type") == "classification":
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

        elif evaluation.get("problem_type") == "regression":
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

        if convergence_warning:
            conclusions["recommendation"] = f"{convergence_warning} {conclusions['recommendation']}"

        return conclusions
