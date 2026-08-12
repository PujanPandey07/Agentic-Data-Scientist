from enum import Enum

from pydantic import BaseModel


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


class VisualizationPlanResponse(BaseModel):
    visualizations: list[VisualizationPlan]
