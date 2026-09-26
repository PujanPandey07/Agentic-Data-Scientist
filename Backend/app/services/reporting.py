import json
import os
import logging
from datetime import datetime, timezone

from services.runtime_state import runtime_state_store

logger = logging.getLogger(__name__)


class ReportingService:
    """Deterministic report builder. Aggregates results from stages that
    actually ran in THIS invocation — never stale data left over in the
    checkpoint from an earlier, unrelated session on the same thread."""

    def generate_report(self, state: dict) -> dict:
        logger.info(
            f"Generating final report for dataset {state.get('dataset_id')}")

        completed = state.get("completed_tasks") or []
        problem_type = self._get_problem_type(state)
        is_clustering = problem_type == "clustering"

        ran_feature_engineering = "feature_engineering" in completed
        # "model_selection"/"hyperparameter_tuning"/"evaluation" are the
        # SAME stage-name strings for both families (see
        # route_task_unsupervised) — which underlying report field to read
        # is decided by problem_type below, not by a different stage name.
        ran_model_selection = "model_selection" in completed
        ran_hyperparameter_tuning = "hyperparameter_tuning" in completed
        ran_evaluation = "evaluation" in completed

        if is_clustering:
            training_section = (
                state.get("clustering_training_report")
                if ran_model_selection else None
            )
            tuning_section = (
                state.get("clustering_tuning_report")
                if ran_hyperparameter_tuning else None
            )
            evaluation_section = (
                state.get("clustering_evaluation_report")
                if ran_evaluation else None
            )
        else:
            training_section = (
                state.get("training_report")
                if ran_model_selection else None
            )
            tuning_section = (
                state.get("hyperparameter_tuning_report")
                if ran_hyperparameter_tuning else None
            )
            evaluation_section = (
                state.get("evaluation_report")
                if ran_evaluation else None
            )

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "user_query": state.get("user_query") or "N/A",
            "dataset_id": state.get("dataset_id") or "N/A",
            "target_column": state.get("target_column") or "N/A",
            "problem_type": problem_type,
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
            "training": training_section,
            "hyperparameter_tuning": tuning_section,
            "evaluation": evaluation_section,
            "conclusions": (
                self._build_clustering_conclusions(
                    state, ran_model_selection, ran_evaluation,
                    training_section, tuning_section, evaluation_section,
                )
                if is_clustering else
                self._build_conclusions(
                    state, ran_model_selection, ran_evaluation)
            ),
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
        if df is None:
            df = runtime_state_store.get_dataset(state.get("dataset_id"))
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

    def _build_clustering_conclusions(
        self, state: dict, ran_model_selection: bool, ran_evaluation: bool,
        training_section: dict | None, tuning_section: dict | None,
        evaluation_section: dict | None,
    ) -> dict:
        """Clustering counterpart to _build_conclusions — no accuracy/F1/R²
        exist here, so this speaks entirely in cluster-count and
        cluster-quality-metric terms instead. Prefers the TUNED result
        (more refined n_clusters/params) when tuning ran, falling back to
        the raw training/model-selection result otherwise — same
        preference order the supervised evaluation stage gives to a tuned
        model over an untuned one."""

        if not ran_model_selection:
            return {
                "best_algorithm": "N/A",
                "n_clusters": "N/A",
                "quality_score": "N/A",
                "recommendation": (
                    "No clustering was part of this run — "
                    "see cleaning/EDA/visualization results above."
                ),
            }

        training = training_section or {}
        tuning = tuning_section or {}

        # Tuning is the more refined result when it ran successfully —
        # same preference the supervised evaluation stage gives a tuned
        # model over the raw training candidate.
        source = tuning if tuning and "error" not in tuning and tuning.get(
            "best_trial_score") is not None else training

        best_algorithm = (
            tuning.get("original_algorithm")
            if source is tuning else training.get("best_algorithm", "N/A")
        )
        n_clusters = source.get("n_clusters_found", "N/A")
        n_noise = source.get("n_noise_points", 0)
        scoring_metric = source.get("scoring_metric", "N/A")
        quality_score = source.get(
            "best_trial_score") if source is tuning else source.get("best_score")

        conclusions = {
            "best_algorithm": best_algorithm or "N/A",
            "n_clusters": n_clusters,
            "n_noise_points": n_noise,
            "scoring_metric": scoring_metric,
            "quality_score": quality_score if quality_score is not None else "N/A",
            "recommendation": "N/A",
        }

        if not ran_evaluation or not evaluation_section or "error" in (evaluation_section or {}):
            conclusions["recommendation"] = (
                "Clustering completed, but a full quality evaluation "
                "did not run or could not be computed in this session."
            )
            return conclusions

        metrics = evaluation_section.get("metrics", {})
        silhouette = metrics.get("silhouette_score")
        conclusions["cluster_sizes"] = evaluation_section.get(
            "cluster_sizes", {})

        if silhouette is not None:
            conclusions["final_silhouette_score"] = silhouette
            if silhouette >= 0.70:
                conclusions["recommendation"] = "Strong, well-separated clusters. Structure is clear and reliable."
            elif silhouette >= 0.50:
                conclusions["recommendation"] = "Reasonable cluster structure. Some overlap between clusters may exist."
            elif silhouette >= 0.25:
                conclusions["recommendation"] = "Weak cluster structure. Consider different features, scaling, or algorithm."
            else:
                conclusions["recommendation"] = "Little to no meaningful cluster structure detected. Data may not be naturally clusterable with the current features."
        else:
            conclusions["recommendation"] = metrics.get(
                "warning",
                "Evaluation ran, but quality metrics could not be computed "
                "(likely fewer than 2 real clusters found).",
            )

        return conclusions
