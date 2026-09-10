from schema.analysis_context import AnalysisContext


class AnalysisContextBuilder:
    """Builds LLM context from completed results, never raw graph state."""

    def build(self, state: dict, report: dict) -> AnalysisContext:
        summary = state.get("dataset_summary")
        dataset = summary.model_dump(mode="json") if summary else {}
        dataset.update({
            "target_column": report.get("target_column"),
            "problem_type": report.get("problem_type"),
            "final_overview": report.get("dataset_overview", {}),
        })

        plan = state.get("model_selection_plan")
        model_selection = plan.model_dump(mode="json") if plan else None

        return AnalysisContext(
            dataset_id=report.get("dataset_id", "N/A"),
            generated_at=report.get("generated_at", ""),
            dataset=dataset,
            cleaning=report.get("cleaning"),
            eda=report.get("eda"),
            visualization=report.get("visualization", {}),
            feature_engineering=report.get("feature_engineering"),
            model_selection=model_selection,
            training=report.get("training"),
            hyperparameter_tuning=report.get("hyperparameter_tuning"),
            evaluation=report.get("evaluation"),
            conclusions=report.get("conclusions", {}),
        )


analysis_context_builder = AnalysisContextBuilder()
