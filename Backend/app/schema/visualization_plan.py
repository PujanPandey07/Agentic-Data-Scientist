from enum import Enum

from pydantic import BaseModel, model_validator


class ChartType(str, Enum):
    BAR = "bar"
    HISTOGRAM = "histogram"
    SCATTER = "scatter"
    BOX = "box"
    HEATMAP = "heatmap"
    LINE = "line"
    VIOLIN = "violin"
    PAIRPLOT = "pairplot"
    PIE = "pie"
    # Clustering-only chart types — never emitted by the LLM planner
    # (visualization_prompt is intentionally left untouched), since the
    # data they need (cluster labels, an n_clusters sweep, a linkage
    # matrix) doesn't exist yet when the planner runs. Generated
    # deterministically instead, later in the clustering pipeline, once
    # that data exists — see VisualizationService.generate_clustering_visualizations.
    CLUSTER_SCATTER = "cluster_scatter"
    ELBOW = "elbow"
    DENDROGRAM = "dendrogram"


class VisualizationPlan(BaseModel):
    chart_type: ChartType
    x_column: str | None = None
    y_column: str | None = None
    group_by: str | None = None
    purpose: str
    priority: int

    @model_validator(mode="after")
    def _validate_required_columns(self):
        """Fail validation (not just execution) when a chart type is
        missing the column(s) it needs. This lets invoke_with_repair's
        validation/retry loop catch and fix it before it ever reaches
        VisualizationService — instead of the whole stage crashing
        deterministically at chart-generation time."""
        if self.chart_type in (ChartType.BAR, ChartType.HISTOGRAM):
            if not self.x_column:
                raise ValueError(
                    f"chart_type='{self.chart_type.value}' requires "
                    f"x_column to be set."
                )
        elif self.chart_type == ChartType.SCATTER:
            if not self.x_column or not self.y_column:
                raise ValueError(
                    "chart_type='scatter' requires both x_column and "
                    "y_column to be set."
                )
        elif self.chart_type == ChartType.BOX:
            if not self.y_column:
                raise ValueError(
                    "chart_type='box' requires y_column to be set."
                )
        elif self.chart_type == ChartType.LINE:
            if not self.x_column or not self.y_column:
                raise ValueError(
                    "chart_type='line' requires both x_column and "
                    "y_column to be set."
                )
        elif self.chart_type == ChartType.VIOLIN:
            if not self.y_column:
                raise ValueError(
                    "chart_type='violin' requires y_column to be set."
                )
        elif self.chart_type == ChartType.PIE:
            if not self.x_column:
                raise ValueError(
                    "chart_type='pie' requires x_column to be set (a "
                    "categorical column with a small number of unique values)."
                )
        # heatmap and pairplot need no explicit column — both use all
        # numeric columns. The three clustering types are never
        # LLM-planned in the first place (see ChartType), so this
        # validator never actually runs against them in practice.
        return self


class VisualizationPlanResponse(BaseModel):
    visualizations: list[VisualizationPlan]
