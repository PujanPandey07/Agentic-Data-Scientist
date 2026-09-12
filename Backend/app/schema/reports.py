# schema/reports.py
from pydantic import BaseModel


class ChartInfo(BaseModel):
    chart_type: str
    x_column: str | None = None
    y_column: str | None = None
    group_by: str | None = None
    url: str  # frontend-usable URL, e.g. /static/charts/<dataset_id>/x.png


class ChartsResponse(BaseModel):
    thread_id: str
    charts: list[ChartInfo]
