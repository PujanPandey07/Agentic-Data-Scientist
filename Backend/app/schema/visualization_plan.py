from enum import Enum

from pydantic import BaseModel, model_validator


class ChartType(str, Enum):
    BAR = "bar"
    HISTOGRAM = "histogram"
    SCATTER = "scatter"
    BOX = "box"
    HEATMAP = "heatmap"


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
        # heatmap needs no explicit column — it uses all numeric columns
        return self


class VisualizationPlanResponse(BaseModel):
    visualizations: list[VisualizationPlan]
